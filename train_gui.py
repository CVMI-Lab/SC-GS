#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_flow
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training_report
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import imageio
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from scipy.spatial.transform import Rotation as R

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GUI:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        if self.opt.progressive_train:
            self.opt.iterations_node_sampling = max(self.opt.iterations_node_sampling, int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio))
            self.opt.iterations_node_rendering = max(self.opt.iterations_node_rendering, self.opt.iterations_node_sampling + 2000)
            print(f'Progressive trian is on. Adjusting the iterations node sampling to {self.opt.iterations_node_sampling} and iterations node rendering {self.opt.iterations_node_rendering}')

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(K=self.dataset.K, deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, skinning=self.args.skinning, hyper_dim=self.dataset.hyper_dim, node_num=self.dataset.node_num, pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color, use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time, d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat, local_frame=self.dataset.local_frame, progressive_brand_time=self.dataset.progressive_brand_time, with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale, enable_densify_prune=self.opt.node_enable_densify_prune, is_scene_static=dataset.is_scene_static)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=-1)
        self.deform.train_setting(opt)

        gs_fea_dim = self.deform.deform.node_num if args.skinning and self.deform.name == 'node' else self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=self.dataset.gs_with_motion_mask)

        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        self.gaussians.training_setup(opt)
        if self.deform.name == 'node' and not deform_loaded:
            if not self.dataset.is_blender:
                if self.opt.random_init_deform_gs:
                    num_pts = 100_000
                    print(f"Generating random point cloud ({num_pts})...")
                    xyz = torch.rand((num_pts, 3)).float().cuda() * 2 - 1
                    mean, scale = self.gaussians.get_xyz.mean(dim=0), self.gaussians.get_xyz.std(dim=0).mean() * 3
                    xyz = xyz * scale + mean
                    self.deform.deform.init(init_pcl=xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=True)
                else:
                    print('Initialize nodes with COLMAP point cloud.')
                    self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=self.dataset.init_isotropic_gs_with_all_colmap_pcl)
            else:
                print('Initialize nodes with Random point cloud.')
                self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=args.skinning)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        self.iteration_node_rendering = 1 if self.scene.loaded_iter is None else self.opt.iterations_node_rendering

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        # For UI
        self.visualization_mode = 'RGB'

        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        self.video_speed = 1.

        # For Animation
        self.animation_time = 0.
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.animation_scaling_bias = None
        self.animate_tool = None
        self.is_training_animation_weight = False
        self.is_training_motion_analysis = False
        self.motion_genmodel = None
        self.motion_animation_d_values = None
        self.showing_overlay = True
        self.should_save_screenshot = False
        self.should_vis_trajectory = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.traj_overlay = None
        self.vis_traj_realtime = False
        self.last_traj_overlay_type = None
        self.view_animation = True
        self.n_rings_N = 2
        # Use ARAP or Generative Model to Deform
        self.deform_mode = "arap_from_init"
        self.should_render_customized_trajectory = False
        self.should_render_customized_trajectory_spiral = False

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def animation_initialize(self, use_traj=True):
        from lap_deform import LapDeform
        gaussians = self.deform.deform.as_gaussians
        fid = torch.tensor(self.animation_time).cuda().float()
        time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
        values = self.deform.deform.node_deform(t=time_input)
        trans = values['d_xyz']
        pcl = gaussians.get_xyz + trans

        if use_traj:
            print('Trajectory analysis!')
            t_samp_num = 16
            t_samp = torch.linspace(0, 1, t_samp_num).cuda().float()
            time_input = t_samp[None, :, None].expand(gaussians.get_xyz.shape[0], -1, 1)
            trajectory = self.deform.deform.node_deform(t=time_input)['d_xyz'] + gaussians.get_xyz[:, None]
        else:
            trajectory = None

        self.animate_init_values = values
        if self.deform.deform.is_scene_static:
            nodes = self.deform.deform.nodes[..., :3]
            scale = torch.norm(nodes.max(0).values - nodes.min(0).values)
            node_radius = scale / 20
            print(f'Static scene node radius: {node_radius}')
        else:
            node_radius = self.deform.deform.node_radius.detach()
        self.animate_tool = LapDeform(init_pcl=pcl, K=4, trajectory=trajectory, node_radius=node_radius)
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        print('Initialize Animation Model with %d control nodes' % len(pcl))

    def animation_reset(self):
        self.animate_tool.reset()
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None
        self.motion_animation_d_values = None
        print('Reset Animation Model ...')

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    def callback_vis_traj_realtime():
                        self.vis_traj_realtime = not self.vis_traj_realtime
                        if not self.vis_traj_realtime:
                            self.traj_coor = None
                        print('Visualize trajectory: ', self.vis_traj_realtime)
                    dpg.add_button(
                        label="Traj",
                        tag="_button_vis_traj",
                        callback=callback_vis_traj_realtime,
                    )
                    dpg.bind_item_theme("_button_vis_traj", theme_button)

                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="NodeMotion",
                        tag="_button_vis_node_motion",
                        callback=callback_vismode,
                        user_data='MotionMask_Node',
                    )
                    dpg.bind_item_theme("_button_vis_node_motion", theme_button)

                    dpg.add_button(
                        label="Node",
                        tag="_button_vis_node",
                        callback=callback_vismode,
                        user_data='Node',
                    )
                    dpg.bind_item_theme("_button_vis_node", theme_button)

                    dpg.add_button(
                        label="Dynamic",
                        tag="_button_vis_Dynamic",
                        callback=callback_vismode,
                        user_data='Dynamic',
                    )
                    dpg.bind_item_theme("_button_vis_Dynamic", theme_button)

                    dpg.add_button(
                        label="Static",
                        tag="_button_vis_Static",
                        callback=callback_vismode,
                        user_data='Static',
                    )
                    dpg.bind_item_theme("_button_vis_Static", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.
                    def callback_speed_control(sender):
                        self.video_speed = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=0.,
                        max_value=3.,
                        min_value=-3.,
                        callback=callback_speed_control,
                    )
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                        self.scene.save(self.iteration)
                        self.deform.save_weights(self.args.model_path, self.iteration)
                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True
                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory = True
                    dpg.add_button(
                        label="render_traj", tag="_button_render_traj", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj", theme_button)

                    def callback_render_traj(sender, app_data):
                        self.should_render_customized_trajectory_spiral = not self.should_render_customized_trajectory_spiral
                        if self.should_render_customized_trajectory_spiral:
                            dpg.configure_item("_button_render_traj_spiral", label="camera")
                        else:
                            dpg.configure_item("_button_render_traj_spiral", label="spiral")
                    dpg.add_button(
                        label="spiral", tag="_button_render_traj_spiral", callback=callback_render_traj
                    )
                    dpg.bind_item_theme("_button_render_traj_spiral", theme_button)
                    
                    def callback_cache_nn(sender, app_data):
                        self.deform.deform.cached_nn_weight = not self.deform.deform.cached_nn_weight
                        print(f'Cached nn weight for higher rendering speed: {self.deform.deform.cached_nn_weight}')
                    dpg.add_button(
                        label="cache_nn", tag="_button_cache_nn", callback=callback_cache_nn
                    )
                    dpg.bind_item_theme("_button_cache_nn", theme_button)

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_save_deform_kpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        self.deform_keypoints.t = self.animation_time
                        save_obj(path=self.args.model_path+'/deform_kpt.pickle', obj=self.deform_keypoints)
                        print('Save kpt done!')
                    dpg.add_button(
                        label="save_deform_kpt", tag="_button_save_deform_kpt", callback=callback_save_deform_kpt
                    )
                    dpg.bind_item_theme("_button_save_deform_kpt", theme_button)

                    def callback_load_deform_kpt(sender, app_data):
                        from utils.pickle_utils import load_obj
                        self.deform_keypoints = load_obj(path=self.args.model_path+'/deform_kpt.pickle')
                        self.animation_time = self.deform_keypoints.t
                        with torch.no_grad():
                            animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), return_R=True)
                            self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                            self.animation_rot_bias = quat
                            self.animation_scaling_bias = ani_d_scaling
                        self.need_update_overlay = True
                        print('Load kpt done!')
                    dpg.add_button(
                        label="load_deform_kpt", tag="_button_load_deform_kpt", callback=callback_load_deform_kpt
                    )
                    dpg.bind_item_theme("_button_load_deform_kpt", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("render", "depth", "alpha", "normal_dep"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
            
            # animation options
            with dpg.collapsing_header(label="Motion Editing", default_open=True):
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Freeze Time: ")
                    def callback_animation_time(sender):
                        self.animation_time = dpg.get_value(sender)
                        self.is_animation = True
                        self.need_update = True
                        # self.animation_initialize()
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=1.,
                        min_value=0.,
                        callback=callback_animation_time,
                    )

                with dpg.group(horizontal=True):
                    def callback_animation_mode(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = not self.is_animation
                            if self.is_animation:
                                if not hasattr(self, 'animate_tool') or self.animate_tool is None:
                                    self.animation_initialize()
                    dpg.add_button(
                        label="Play",
                        tag="_button_vis_animation",
                        callback=callback_animation_mode,
                        user_data='Animation',
                    )
                    dpg.bind_item_theme("_button_vis_animation", theme_button)

                    def callback_animation_initialize(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_initialize()
                    dpg.add_button(
                        label="Init Graph",
                        tag="_button_init_graph",
                        callback=callback_animation_initialize,
                    )
                    dpg.bind_item_theme("_button_init_graph", theme_button)

                    def callback_clear_animation(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = True
                            self.animation_reset()
                    dpg.add_button(
                        label="Clear Graph",
                        tag="_button_clc_animation",
                        callback=callback_clear_animation,
                    )
                    dpg.bind_item_theme("_button_clc_animation", theme_button)

                    def callback_overlay(sender, app_data):
                        if self.showing_overlay:
                            self.showing_overlay = False
                            dpg.configure_item("_button_train_motion_gen", label="show overlay")
                        else:
                            self.showing_overlay = True
                            dpg.configure_item("_button_train_motion_gen", label="close overlay")                    
                    dpg.add_button(
                        label="close overlay", tag="_button_overlay", callback=callback_overlay
                    )
                    dpg.bind_item_theme("_button_overlay", theme_button)

                    def callback_save_ckpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        if not self.is_animation:
                            print('Please switch to animation mode!')
                        deform_keypoint_files = sorted([file for file in os.listdir(os.path.join(self.args.model_path)) if file.startswith('deform_keypoints') and file.endswith('.pickle')])
                        if len(deform_keypoint_files) > 0:
                            newest_id = int(deform_keypoint_files[-1].split('.')[0].split('_')[-1])
                        else:
                            newest_id = -1
                        save_obj(os.path.join(self.args.model_path, f'deform_keypoints_{newest_id+1}.pickle'), [self.deform_keypoints, self.animation_time])
                    dpg.add_button(
                        label="sv_kpt", tag="_button_save_kpt", callback=callback_save_ckpt
                    )
                    dpg.bind_item_theme("_button_save_kpt", theme_button)

                    def callback_load_ckpt(sender, app_data):
                        from utils.pickle_utils import load_obj
                        if not hasattr(self, 'deform_kpt_files') or self.deform_kpt_files is None:
                            self.deform_kpt_files = sorted([file for file in os.listdir(os.path.join(self.args.model_path)) if file.startswith('deform_keypoints') and file.endswith('.pickle')])
                            self.deform_kpt_files_idx = 0
                        else:
                            self.deform_kpt_files_idx = (self.deform_kpt_files_idx + 1) % len(self.deform_kpt_files)
                        print(f'Load {self.deform_kpt_files[self.deform_kpt_files_idx]}')
                        deform_keypoints, self.animation_time = load_obj(os.path.join(self.args.model_path, self.deform_kpt_files[self.deform_kpt_files_idx]))
                        self.is_animation = True
                        self.animation_initialize()
                        animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=deform_keypoints.get_kpt_idx(), handle_pos=deform_keypoints.get_deformed_kpt_np(), return_R=True)
                        self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                        self.animation_rot_bias = quat
                        self.animation_scaling_bias = ani_d_scaling
                        self.update_control_point_overlay()
                    dpg.add_button(
                        label="ld_kpt", tag="_button_load_kpt", callback=callback_load_ckpt
                    )
                    dpg.bind_item_theme("_button_load_kpt", theme_button)

                with dpg.group(horizontal=True):
                    def callback_change_deform_mode(sender, app_data):
                        self.deform_mode = app_data
                        self.need_update = True
                    dpg.add_combo(
                        ("arap_iterative", "arap_from_init"),
                        label="Editing Mode",
                        default_value=self.deform_mode,
                        callback=callback_change_deform_mode,
                    )

                with dpg.group(horizontal=True):
                    def callback_change_n_rings_N(sender, app_data):
                        self.n_rings_N = int(app_data)
                    dpg.add_combo(
                        ("0", "1", "2", "3", "4"),
                        label="n_rings",
                        default_value="2",
                        callback=callback_change_n_rings_N,
                    )
                    

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_drag(sender, app_data):
            if not self.is_animation:
                print("Please switch to animation mode!")
                return
            if not dpg.is_item_focused("_primary_window"):
                return
            if len(self.deform_keypoints.get_kpt()) == 0:
                return
            if self.animate_tool is None:
                self.animation_initialize()
            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
            if dpg.is_key_down(dpg.mvKey_R):
                side = self.cam.rot.as_matrix()[:3, 0]
                up = self.cam.rot.as_matrix()[:3, 1]
                forward = self.cam.rot.as_matrix()[:3, 2]
                rotvec_z = forward * np.radians(-0.05 * dx)
                rotvec_y = up * np.radians(-0.05 * dy)
                rot_mat = (R.from_rotvec(rotvec_z)).as_matrix() @ (R.from_rotvec(rotvec_y)).as_matrix()
                self.deform_keypoints.set_rotation_delta(rot_mat)
            else:
                delta = 0.00010 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])
                self.deform_keypoints.update_delta(delta)
                self.need_update_overlay = True

            if self.deform_mode.startswith("arap"):
                with torch.no_grad():
                    if self.deform_mode == "arap_from_init" or self.animation_trans_bias is None:
                        init_verts = None
                    else:
                        init_verts = self.animation_trans_bias + self.animate_tool.init_pcl
                    animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), init_verts=init_verts, return_R=True)
                    self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                    self.animation_rot_bias = quat
                    self.animation_scaling_bias = ani_d_scaling

        def callback_keypoint_add(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            ##### select keypoints by shift + click
            if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_D) or dpg.is_key_down(dpg.mvKey_F) or dpg.is_key_down(dpg.mvKey_A) or dpg.is_key_down(dpg.mvKey_Q):
                if not self.is_animation:
                    print("Please switch to animation mode!")
                    return
                # Rendering the image with node gaussians to select nodes as keypoints
                fid = torch.tensor(self.animation_time).cuda().float()
                cur_cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    fid = fid
                )
                with torch.no_grad():
                    time_input = self.deform.deform.expand_time(fid)
                    d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, node_trans_bias=self.animation_trans_bias, node_rot_bias=self.animation_rot_bias, node_scaling_bias=self.animation_scaling_bias)
                    gaussians = self.gaussians
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']

                    out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])

                    d = out['depth'][0][ph, pw]
                    z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * d - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                    uvz = torch.tensor([((pw-.5)/self.W * 2 - 1) * d, ((ph-.5)/self.H*2-1) * d, z, d]).cuda().float().view(1, 4)
                    p3d = (uvz @ torch.inverse(cur_cam.full_proj_transform))[0, :3]

                    # Pick the closest node as the keypoint
                    node_trans = self.deform.deform.node_deform(time_input)['d_xyz']
                    if self.animation_trans_bias is not None:
                        node_trans = node_trans + self.animation_trans_bias
                    nodes = self.deform.deform.nodes[..., :3] + node_trans
                    keypoint_idxs = torch.tensor([(p3d - nodes).norm(dim=-1).argmin()]).cuda()

                if dpg.is_key_down(dpg.mvKey_A):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs)
                    print(f'Add kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_S):
                    self.deform_keypoints.select_kpt(keypoint_idxs.item())

                elif dpg.is_key_down(dpg.mvKey_D):
                    if True:
                        keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=self.n_rings_N)
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Expand kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_F):
                    keypoint_idxs = torch.arange(nodes.shape[0]).cuda()
                    keypoint_3ds = nodes[keypoint_idxs]
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                    print(f'Add all the control points as kpt: {self.deform_keypoints.selective_keypoints_idx_list}')

                elif dpg.is_key_down(dpg.mvKey_Q):
                    self.deform_keypoints.select_rotation_kpt(keypoint_idxs.item())
                    print(f"select rotation control points: {keypoint_idxs.item()}")

                self.need_update_overlay = True

        self.callback_keypoint_add = callback_keypoint_add
        self.callback_keypoint_drag = callback_keypoint_drag

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_drag)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_keypoint_add)

        dpg.create_viewport(
            title="SC-GS",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()
    
    @torch.no_grad()
    def draw_gs_trajectory(self, time_gap=0.3, samp_num=512, gs_num=512, thickness=1):
        fid = torch.tensor(self.animation_time).cuda().float() if self.is_animation else torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)
        from utils.pickle_utils import load_obj, save_obj
        if os.path.exists(os.path.join(self.args.model_path, 'trajectory_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'trajectory_camera.pickle'))
            cur_cam = load_obj(os.path.join(self.args.model_path, 'trajectory_camera.pickle'))
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
            save_obj(os.path.join(self.args.model_path, 'trajectory_camera.pickle'), cur_cam)
        fid = cur_cam.fid
        
        # Calculate the gs position at t0
        t = fid
        time_input = t.unsqueeze(0).expand(self.gaussians.get_xyz.shape[0], -1) if self.deform.name == 'mlp' else self.deform.deform.expand_time(t)
        d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, motion_mask=self.gaussians.motion_mask)
        cur_pts = self.gaussians.get_xyz + d_values['d_xyz']
    
        if not os.path.exists(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle')):
            from utils.time_utils import farthest_point_sample
            pts_idx = farthest_point_sample(cur_pts[None], gs_num)[0]
            save_obj(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'), cur_pts[pts_idx].detach().cpu().numpy())
        else:
            print('Load keypoints from ', os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'))
            kpts = torch.from_numpy(load_obj(os.path.join(self.args.model_path, 'trajectory_keypoints.pickle'))).cuda()
            import pytorch3d.ops
            _, idxs, _ = pytorch3d.ops.knn_points(kpts[None], cur_pts[None], None, None, K=1)
            pts_idx = idxs[0,:,0]
        delta_ts = torch.linspace(0, time_gap, samp_num)
        traj_pts = []
        for i in range(samp_num):
            t = fid + delta_ts[i]
            time_input = t.unsqueeze(0).expand(gs_num, -1) if self.deform.name == 'mlp' else self.deform.deform.expand_time(t)
            d_values = self.deform.step(self.gaussians.get_xyz[pts_idx].detach(), time_input, feature=self.gaussians.feature[pts_idx], is_training=False, motion_mask=self.gaussians.motion_mask[pts_idx])
            cur_pts = self.gaussians.get_xyz[pts_idx] + d_values['d_xyz']
            cur_pts = torch.cat([cur_pts, torch.ones_like(cur_pts[..., :1])], dim=-1)
            cur_pts2d = cur_pts @ cur_cam.full_proj_transform
            cur_pts2d = cur_pts2d[..., :2] / cur_pts2d[..., -1:]
            cur_pts2d = (cur_pts2d + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width]).cuda()
            traj_pts.append(cur_pts2d)
        traj_pts = torch.stack(traj_pts, dim=1).detach().cpu().numpy()  # N, T, 2

        import cv2
        from matplotlib import cm
        color_map = cm.get_cmap("jet")
        colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
        traj_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1) * 255
        Image.fromarray(traj_img.astype('uint8')).save(os.path.join(self.args.model_path, 'trajectory.png'))
        
        from utils.vis_utils import render_cur_cam
        img_begin = render_cur_cam(self=self, cur_cam=cur_cam)
        cur_cam.fid = cur_cam.fid + delta_ts[-1]
        img_end = render_cur_cam(self=self, cur_cam=cur_cam)
        img_begin = (img_begin.permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        img_end = (img_end.permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(img_begin).save(os.path.join(self.args.model_path, 'traj_start.png'))
        Image.fromarray(img_end).save(os.path.join(self.args.model_path, 'traj_end.png'))

    # gui mode
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step()
                else:
                    self.train_step()
            if self.should_vis_trajectory:
                self.draw_gs_trajectory()
                self.should_vis_trajectory = False
            if self.should_render_customized_trajectory:
                self.render_customized_trajectory(use_spiral=self.should_render_customized_trajectory_spiral)
            self.test_step()

            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step()
                else:
                    self.train_step()
    
    def train_step(self):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.do_shs_python, self.pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
                if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            if self.opt.progressive_train and self.iteration < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio):
                cameras_to_train_idx = int(min(((self.iteration) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                cameras_to_train_idx = max(cameras_to_train_idx, 1)
                interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                min_idx = max(0, cameras_to_train_idx - interval_len)
                sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                out_domain_idx = np.arange(min_idx)
                if len(out_domain_idx) >= interval_len:
                    out_domain_idx = np.random.choice(out_domain_idx, [interval_len], replace=False)
                    out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                    viewpoint_stack = viewpoint_stack + out_domain_stack
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack
        
        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if self.deform.name == 'mlp' or self.deform.name == 'static':
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, camera_center=viewpoint_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        elif self.deform.name == 'node':
            if not self.deform.deform.inited:
                print('Notice that warping nodes are initialized with Gaussians!!!')
                self.deform.deform.init(self.opt, self.gaussians.get_xyz.detach(), feature=self.gaussians.feature)
            time_input = self.deform.deform.expand_time(fid)
            N = time_input.shape[0]
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam.camera_center, time_interval=time_interval)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_xyz.detach(), d_rotation.detach(), d_scaling.detach(), d_opacity.detach() if d_opacity is not None else None, d_color.detach() if d_color is not None else None
            elif self.iteration < self.opt.dynamic_color_warm_up:
                d_color = d_color.detach() if d_color is not None else None

        # Render
        random_bg_color = (not self.dataset.white_background and self.opt.random_bg_color) and self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if random_bg_color:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img

        if self.iteration > self.opt.warm_up:
            loss = loss + self.deform.reg_loss

        # Flow loss
        flow_id2_candidates = viewpoint_cam.flow_dirs
        lambda_optical = landmark_interpolate(self.opt.lambda_optical_landmarks, self.opt.lambda_optical_steps, self.iteration)
        if flow_id2_candidates != [] and lambda_optical > 0 and self.iteration >= self.opt.warm_up:
            # Pick flow file and read it
            flow_id2_dir = np.random.choice(flow_id2_candidates)
            flow = np.load(flow_id2_dir)
            mask_id2_dir = flow_id2_dir.replace('raft_neighbouring', 'raft_masks').replace('.npy', '.png')
            masks = imageio.imread(mask_id2_dir) / 255.
            flow = torch.from_numpy(flow).float().cuda()
            masks = torch.from_numpy(masks).float().cuda()
            if flow.shape[0] != image.shape[1] or flow.shape[1] != image.shape[2]:
                flow = torch.nn.functional.interpolate(flow.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
                masks = torch.nn.functional.interpolate(masks.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
            fid1 = viewpoint_cam.fid
            cam2_id = os.path.basename(flow_id2_dir).split('_')[-1].split('.')[0]
            if not hasattr(self, 'img2cam'):
                self.img2cam = {cam.image_name: idx for idx, cam in enumerate(self.scene.getTrainCameras().copy())}
            if cam2_id in self.img2cam:  # Only considering the case with existing files
                cam2_id = self.img2cam[cam2_id]
                viewpoint_cam2 = self.scene.getTrainCameras().copy()[cam2_id]
                fid2 = viewpoint_cam2.fid
                # Calculate the GT flow, weight, and mask
                coor1to2_flow = flow / torch.tensor(flow.shape[:2][::-1], dtype=torch.float32).cuda() * 2
                cycle_consistency_mask = masks[..., 0] > 0
                occlusion_mask = masks[..., 1] > 0
                mask_flow = cycle_consistency_mask | occlusion_mask
                pair_weight = torch.clamp(torch.cos((fid1 - fid2).abs() * np.pi / 2), 0.2, 1)
                # Calculate the motion at t2
                time_input2 = self.deform.deform.expand_time(fid2)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_xyz2 = self.deform.step(self.gaussians.get_xyz.detach(), time_input2 + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam2.camera_center)['d_xyz']
                # Render the flow image
                render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint_cam, viewpoint_camera2=viewpoint_cam2, d_xyz1=d_xyz, d_xyz2=d_xyz2, d_rotation1=d_rotation, d_scaling1=d_scaling, scale_const=None)
                coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                mask_motion = (render_pkg2['alpha'][0] > .9).detach()  # Only optimizing the space with solid points to avoid dilation
                mask = (mask_motion & mask_flow)[..., None] * pair_weight
                # Flow loss based on pixel rgb loss
                l1_loss_weight = (image.detach() - gt_image).abs().mean(dim=0)
                l1_loss_weight = torch.cos(l1_loss_weight * torch.pi / 2)
                mask = mask * l1_loss_weight[..., None]
                # Flow mask
                optical_flow_loss = l1_loss(mask * coor1to2_flow, mask * coor1to2_motion)
                loss = loss + lambda_optical * optical_flow_loss

        # Motion Mask Loss
        lambda_motion_mask = landmark_interpolate(self.opt.lambda_motion_mask_landmarks, self.opt.lambda_motion_mask_steps, self.iteration)
        if not self.opt.no_motion_mask_loss and self.deform.name == 'node' and self.opt.gt_alpha_mask_as_dynamic_mask and viewpoint_cam.gt_alpha_mask is not None and lambda_motion_mask > 0:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            render_pkg_motion = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=True, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
            motion_image = render_pkg_motion["render"][0]
            L_motion = l1_loss(gt_alpha_mask, motion_image)
            loss = loss + L_motion * lambda_motion_mask

        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, render, (self.pipe, self.background), self.deform, self.dataset.load2gpu_on_the_fly, self.progress_bar)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.node_densify_from_iter and self.iteration % self.opt.node_densification_interval == 0 and self.iteration < self.opt.node_densify_until_iter and self.iteration > self.opt.warm_up or self.iteration == self.opt.node_force_densify_prune_step:
                    # Nodes densify
                    self.deform.densify(max_grad=self.opt.densify_grad_threshold, x=self.gaussians.get_xyz, x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom, feature=self.gaussians.feature, force_dp=(self.iteration == self.opt.node_force_densify_prune_step))

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)
                
        self.deform.update(max(0, self.iteration - self.opt.warm_up))

        if self.gui:
            dpg.set_value(
                "_log_train_psnr",
                "Best PSNR={} in Iteration {}, SSIM={}, LPIPS={},\n MS-SSIM={}, Alex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips)
            )
        else:
            self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
            )
   
    def train_node_rendering_step(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            if self.opt.progressive_train_node and self.iteration_node_rendering < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio) + self.opt.node_warm_up:
                if self.iteration_node_rendering < self.opt.node_warm_up:
                    sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                    max_cam_num = max(30, int(0.01 * len(sorted_train_cams)))
                    viewpoint_stack = sorted_train_cams[0: max_cam_num]
                else:
                    cameras_to_train_idx = int(min(((self.iteration_node_rendering - self.opt.node_warm_up) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                    cameras_to_train_idx = max(cameras_to_train_idx, 1)
                    interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                    min_idx = max(0, cameras_to_train_idx - interval_len)
                    sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                    viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                    out_domain_idx = np.concatenate([np.arange(min_idx), np.arange(cameras_to_train_idx, min(len(self.scene.getTrainCameras()), cameras_to_train_idx+interval_len))])
                    if len(out_domain_idx) >= interval_len:
                        out_domain_len = min(interval_len*5, len(out_domain_idx))
                        out_domain_idx = np.random.choice(out_domain_idx, [out_domain_len], replace=False)
                        out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                        viewpoint_stack = viewpoint_stack + out_domain_stack
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        
        time_input = fid.unsqueeze(0).expand(self.deform.deform.as_gaussians.get_xyz.shape[0], -1)
        N = time_input.shape[0]

        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame

        ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration_node_rendering)
        d_values = self.deform.deform.query_network(x=self.deform.deform.as_gaussians.get_xyz.detach(), t=time_input + ast_noise)
        d_xyz, d_opacity, d_color = d_values['d_xyz'] * self.deform.deform.as_gaussians.motion_mask, d_values['d_opacity'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_color'] is not None else None
        d_rot, d_scale = 0., 0.
        if self.iteration_node_rendering < self.opt.node_warm_up:
            d_xyz = d_xyz.detach()
        d_color = d_color.detach() if d_color is not None else None
        d_opacity = d_opacity.detach() if d_opacity is not None else None

        # Render
        random_bg_color = (self.opt.gt_alpha_mask_as_scene_mask or (self.opt.gt_alpha_mask_as_dynamic_mask and not self.deform.deform.as_gaussians.with_motion_mask)) and viewpoint_cam.gt_alpha_mask is not None
        render_pkg_re = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if random_bg_color:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + render_pkg_re['bg_color'][:, None, None] * (1 - gt_alpha_mask)
        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img

        if self.iteration_node_rendering > self.opt.node_warm_up:
            if not self.deform.deform.use_hash:
                elastic_loss = 1e-3 * self.deform.deform.elastic_loss(t=fid, delta_t=time_interval)
                loss_acc = 1e-5 * self.deform.deform.acc_loss(t=fid, delta_t=3*time_interval)
                loss = loss + elastic_loss + loss_acc
            if not self.opt.no_arap_loss:
                loss_opt_trans = 1e-2 * self.deform.deform.arap_loss()
                loss = loss + loss_opt_trans

        # Motion Mask Loss
        if self.opt.gt_alpha_mask_as_dynamic_mask and self.deform.deform.as_gaussians.with_motion_mask and viewpoint_cam.gt_alpha_mask is not None and self.iteration_node_rendering > self.opt.node_warm_up:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()[0]
            render_pkg_motion = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=self.deform.deform.as_gaussians.with_motion_mask, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
            motion_image = render_pkg_motion["render"][0]
            L_motion = l1_loss(gt_alpha_mask, motion_image)
            loss = loss + L_motion

        loss.backward()
        with torch.no_grad():
            if self.iteration_node_rendering < self.opt.iterations_node_sampling:
                # Densification
                self.deform.deform.as_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if self.iteration_node_rendering % self.opt.densification_interval == 0 or self.iteration_node_rendering == self.opt.node_warm_up - 1:
                    size_threshold = 20 if self.iteration_node_rendering > self.opt.opacity_reset_interval else None
                    if self.dataset.is_blender:
                        grad_max = self.opt.densify_grad_threshold
                    else:
                        if self.deform.deform.as_gaussians.get_xyz.shape[0] > self.deform.deform.node_num * self.opt.node_max_num_ratio_during_init:
                            grad_max = torch.inf
                        else:
                            grad_max = self.opt.densify_grad_threshold
                    self.deform.deform.as_gaussians.densify_and_prune(grad_max, 0.005, self.scene.cameras_extent, size_threshold)
                if self.iteration_node_rendering % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration_node_rendering == self.opt.densify_from_iter):
                    self.deform.deform.as_gaussians.reset_opacity()
            elif self.iteration_node_rendering == self.opt.iterations_node_sampling:
                # Downsampling nodes for sparse control
                # Strategy 1: Directly use the original gs as nodes
                # Strategy 2: Sampling in the hyper space across times
                strategy = self.opt.deform_downsamp_strategy
                if strategy == 'direct':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    self.deform.deform.init(opt=self.opt, init_pcl=original_gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc)
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest)
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling)
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity)
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation)
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature)
                    print('Reset the optimizer of the deform model.')
                    self.deform.train_setting(self.opt)
                elif strategy == 'samp_hyper':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    time_num = 16
                    t_samp = torch.linspace(0, 1, time_num).cuda()
                    x = original_gaussians.get_xyz.detach()
                    trans_samp = []
                    for i in range(time_num):
                        time_input = t_samp[i:i+1, None].expand_as(x[..., :1])
                        trans_samp.append(self.deform.deform.query_network(x=x, t=time_input)['d_xyz'] * original_gaussians.motion_mask)
                    trans_samp = torch.stack(trans_samp, dim=1)
                    hyper_pcl = (trans_samp + original_gaussians.get_xyz[:, None]).reshape([original_gaussians.get_xyz.shape[0], -1])
                    dynamic_mask = self.deform.deform.as_gaussians.motion_mask[..., 0] > .5
                    if not self.opt.deform_downsamp_with_dynamic_mask:
                        dynamic_mask = torch.ones_like(dynamic_mask)
                    idx = self.deform.deform.init(init_pcl=original_gaussians.get_xyz[dynamic_mask], hyper_pcl=hyper_pcl[dynamic_mask], force_init=True, opt=self.opt, reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc[dynamic_mask][idx])
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest[dynamic_mask][idx])
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling[dynamic_mask][idx])
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity[dynamic_mask][idx])
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation[dynamic_mask][idx])
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature[dynamic_mask][idx])
                    gaussians.training_setup(self.opt)
                # No update at the step
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.zero_grad()

            if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1 and self.iteration_node_rendering > self.opt.iterations_node_sampling:
                # Just finish node training and has down sampled control nodes
                self.deform.deform.nodes.data[..., :3] = self.deform.deform.as_gaussians._xyz

            if not self.iteration_node_rendering == self.opt.iterations_node_sampling and not self.iteration_node_rendering == self.opt.iterations_node_rendering - 1:
                # Optimizer step
                self.deform.deform.as_gaussians.optimizer.step()
                self.deform.deform.as_gaussians.update_learning_rate(self.iteration_node_rendering)
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.update_learning_rate(self.iteration_node_rendering)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
        
        self.deform.update(max(0, self.iteration_node_rendering - self.opt.node_warm_up))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        self.iteration_node_rendering += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration_node_rendering: 5d} loss = {loss.item():.4f}",
            )
    
    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10
        
        if self.is_animation:
            if not self.showing_overlay:
                self.buffer_overlay = None
            else:
                self.update_control_point_overlay()
            fid = torch.tensor(self.animation_time).cuda().float()
        else:
            fid = torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)

        if self.should_save_screenshot and os.path.exists(os.path.join(self.args.model_path, 'screenshot_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
            from utils.pickle_utils import load_obj
            cur_cam = load_obj(os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
        elif specified_cam is not None:
            cur_cam = specified_cam
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
        fid = cur_cam.fid

        if self.deform.name == 'node':
            if 'Node' in self.visualization_mode:
                d_rotation_bias = None
                gaussians = self.deform.deform.as_gaussians
                time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
                d_values = self.deform.deform.query_network(x=gaussians.get_xyz.detach(), t=time_input)
                if self.motion_animation_d_values is not None:
                    for key in self.motion_animation_d_values:
                        d_values[key] = self.motion_animation_d_values[key]
                d_xyz, d_opacity, d_color = d_values['d_xyz'] * gaussians.motion_mask, d_values['d_opacity'] * gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * gaussians.motion_mask if d_values['d_color'] is not None else None
                d_rotation, d_scaling = 0., 0.
                if self.view_animation and self.animation_trans_bias is not None:
                    d_xyz = d_xyz + self.animation_trans_bias
                vis_scale_const = self.vis_scale_const
            else:
                if self.view_animation:
                    node_trans_bias = self.animation_trans_bias
                else:
                    node_trans_bias = None
                time_input = self.deform.deform.expand_time(fid)
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, node_trans_bias=node_trans_bias, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, animation_d_values=self.motion_animation_d_values)
                gaussians = self.gaussians
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
                vis_scale_const = None
                d_rotation_bias = d_values['d_rotation_bias'] if 'd_rotation_bias' in d_values.keys() else None
        else:
            vis_scale_const = None
            d_rotation_bias = None
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
                gaussians = self.gaussians
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                gaussians = self.gaussians
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, camera_center=cur_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        
        if self.vis_traj_realtime:
            if 'Node' in self.visualization_mode:
                if self.last_traj_overlay_type != 'node':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam, gs_num=512)
                self.last_traj_overlay_type = 'node'
            else:
                if self.last_traj_overlay_type != 'gs':
                    self.traj_coor = None
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz+d_xyz, camera=cur_cam)
                self.last_traj_overlay_type = 'gs'
        
        if self.visualization_mode == 'Dynamic' or self.visualization_mode == 'Static':
            d_opacity = torch.zeros_like(self.gaussians.motion_mask)
            if self.visualization_mode == 'Dynamic':
                d_opacity[self.gaussians.motion_mask < .9] = - 1e3
            else:
                d_opacity[self.gaussians.motion_mask > .1] = - 1e3
        
        render_motion = "Motion" in self.visualization_mode
        if render_motion:
            vis_scale_const = self.vis_scale_const
        if type(d_rotation) is not float and gaussians._rotation.shape[0] != d_rotation.shape[0]:
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            print('Async in Gaussian Switching')
        out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, render_motion=render_motion, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res, scale_const=vis_scale_const, d_rotation_bias=d_rotation_bias)

        if self.mode == "normal_dep":
            from utils.other_utils import depth2normal
            normal = depth2normal(out["depth"])
            out["normal_dep"] = (normal + 1) / 2

        buffer_image = out[self.mode]  # [3, H, W]

        if self.should_save_screenshot:
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0,1).permute(1,2,0).detach().cpu().numpy()
            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len(os.listdir(image_dir))
                print('>>> Saving image to %s' % os.path.join(image_dir, '%05d.png'%idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%05d.png'%idx))
                # Save the camera of screenshot
                from utils.pickle_utils import save_obj
                save_obj(os.path.join(image_dir, '%05d_cam.pickle'% idx), cur_cam)
            save_image(sv_image, self.screenshot_sv_path)
            self.should_save_screenshot = False

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.is_animation and self.buffer_overlay is not None:
            overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
            try:
                buffer_image = self.buffer_image * overlay_mask + self.buffer_overlay
            except:
                buffer_image = self.buffer_image
        else:
            buffer_image = self.buffer_image

        if self.vis_traj_realtime:
            buffer_image = buffer_image * (1 - self.traj_overlay[..., 3:]) + self.traj_overlay[..., :3] * self.traj_overlay[..., 3:]

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
            dpg.set_value(
                "_texture", buffer_image
            )  # buffer must be contiguous, else seg fault!
        return buffer_image

    def update_control_point_overlay(self):
        from skimage.draw import line_aa
        # should update overlay
        # if self.need_update_overlay and len(self.keypoint_3ds) > 0:
        if self.need_update_overlay and len(self.deform_keypoints.get_kpt()) > 0:
            try:
                buffer_overlay = np.zeros_like(self.buffer_image)
                mv = self.cam.view # [4, 4]
                proj = self.cam.perspective # [4, 4]
                mvp = proj @ mv
                # do mvp transform for keypoints
                # source_points = np.array(self.keypoint_3ds)
                source_points = np.array(self.deform_keypoints.get_kpt())
                # target_points = source_points + np.array(self.keypoint_3ds_delta)
                target_points = self.deform_keypoints.get_deformed_kpt_np()
                points_indices = np.arange(len(source_points))

                source_points_clip = np.matmul(np.pad(source_points, ((0, 0), (0, 1)), constant_values=1.0), mvp.T)  # [N, 4]
                target_points_clip = np.matmul(np.pad(target_points, ((0, 0), (0, 1)), constant_values=1.0), mvp.T)  # [N, 4]
                source_points_clip[:, :3] /= source_points_clip[:, 3:] # perspective division
                target_points_clip[:, :3] /= target_points_clip[:, 3:] # perspective division

                source_points_2d = (((source_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(np.int32)
                target_points_2d = (((target_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(np.int32)

                radius = int((self.H + self.W) / 2 * 0.005)
                keypoint_idxs_to_drag = self.deform_keypoints.selective_keypoints_idx_list
                for i in range(len(source_points_clip)):
                    point_idx = points_indices[i]
                    # draw source point
                    if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < self.W - radius and source_points_2d[i, 1] >= radius and source_points_2d[i, 1] < self.H - radius:
                        buffer_overlay[source_points_2d[i, 1]-radius:source_points_2d[i, 1]+radius, source_points_2d[i, 0]-radius:source_points_2d[i, 0]+radius] += np.array([1,0,0]) if not point_idx in keypoint_idxs_to_drag else np.array([1,0.87,0])
                        # draw target point
                        if target_points_2d[i, 0] >= radius and target_points_2d[i, 0] < self.W - radius and target_points_2d[i, 1] >= radius and target_points_2d[i, 1] < self.H - radius:
                            buffer_overlay[target_points_2d[i, 1]-radius:target_points_2d[i, 1]+radius, target_points_2d[i, 0]-radius:target_points_2d[i, 0]+radius] += np.array([0,0,1]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,0.5,1])
                        # draw line
                        rr, cc, val = line_aa(source_points_2d[i, 1], source_points_2d[i, 0], target_points_2d[i, 1], target_points_2d[i, 0])
                        in_canvas_mask = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
                        buffer_overlay[rr[in_canvas_mask], cc[in_canvas_mask]] += val[in_canvas_mask, None] * np.array([0,1,0]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,1,0])
                self.buffer_overlay = buffer_overlay
            except:
                print('Async Fault in Overlay Drawing!')
                self.buffer_overlay = None

    def update_trajectory_overlay(self, gs_xyz, camera, samp_num=32, gs_num=512, thickness=1):
        if not hasattr(self, 'traj_coor') or self.traj_coor is None:
            from utils.time_utils import farthest_point_sample
            self.traj_coor = torch.zeros([0, gs_num, 4], dtype=torch.float32).cuda()
            opacity_mask = self.gaussians.get_opacity[..., 0] > .1 if self.gaussians.get_xyz.shape[0] == gs_xyz.shape[0] else torch.ones_like(gs_xyz[:, 0], dtype=torch.bool)
            masked_idx = torch.arange(0, opacity_mask.shape[0], device=opacity_mask.device)[opacity_mask]
            self.traj_idx = masked_idx[farthest_point_sample(gs_xyz[None, opacity_mask], gs_num)[0]]
            from matplotlib import cm
            self.traj_color_map = cm.get_cmap("jet")
        pts = gs_xyz[None, self.traj_idx]
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
        self.traj_coor = torch.cat([self.traj_coor, pts], axis=0)
        if self.traj_coor.shape[0] > samp_num:
            self.traj_coor = self.traj_coor[-samp_num:]
        traj_uv = self.traj_coor @ camera.full_proj_transform
        traj_uv = traj_uv[..., :2] / traj_uv[..., -1:]
        traj_uv = (traj_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        traj_uv = traj_uv.detach().cpu().numpy()

        import cv2
        colors = np.array([np.array(self.traj_color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        traj_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1)
        self.traj_overlay = traj_img
        
    def test_speed(self, round=500):
        self.deform.deform.cached_nn_weight = True
        self.test_step()
        t0 = time.time()
        for i in range(round):
            self.test_step()
        t1 = time.time()
        fps = round / (t1 - t0)
        print(f'FPS: {fps}')
        return fps
    
    def render_customized_trajectory(self, use_spiral=False, traj_dir=None, fps=30, motion_repeat=1):
        from utils.pickle_utils import load_obj
        # Remove history trajectory
        if self.vis_traj_realtime:
            self.traj_coor = None
            self.traj_overlay = None
        # Default trajectory path
        if traj_dir is None:
            traj_dir = os.path.join(self.args.model_path, 'trajectory')
        # Read deformation files for animation presentation
        deform_keypoint_files = [None] + sorted([file for file in os.listdir(os.path.join(self.args.model_path)) if file.startswith('deform_keypoints') and file.endswith('.pickle')])
        rendering_animation = len(deform_keypoint_files) > 1
        if rendering_animation:
            deform_keypoints, self.animation_time = load_obj(os.path.join(self.args.model_path, deform_keypoint_files[1]))
            self.animation_initialize()
        # Read camera trajectory files
        if os.path.exists(traj_dir):
            cameras = sorted([cam for cam in os.listdir(traj_dir) if cam.endswith('.pickle')])
            cameras = [load_obj(os.path.join(traj_dir, cam)) for cam in cameras]
            if len(cameras) < 2:
                print('No trajectory cameras found')
                self.should_render_customized_trajectory = False
                return
            if os.path.exists(os.path.join(traj_dir, 'time.txt')):
                with open(os.path.join(traj_dir, 'time.txt'), 'r') as file:
                    time = file.readline()
                    time = time.split(' ')
                    timesteps = np.array([float(t) for t in time])
            else:
                timesteps = np.array([3] * len(cameras))  # three seconds by default
        elif use_spiral:
            from utils.pose_utils import render_path_spiral
            from copy import deepcopy
            c2ws = []
            for camera in self.scene.getTrainCameras():
                c2w = np.eye(4)
                c2w[:3, :3] = camera.R
                c2w[:3, 3] = camera.T
                c2ws.append(c2w)
            c2ws = np.stack(c2ws, axis=0)
            poses = render_path_spiral(c2ws=c2ws, focal=self.cam.fovx*200, rots=3, N=30*12)
            print(f'Use spiral camera poses with {poses.shape[0]} cameras!')
            cameras_ = []
            for i in range(len(poses)):
                cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    0
                )
                cam.reset_extrinsic(R=poses[i, :3, :3], T=poses[i, :3, 3])
                cameras_.append(cam)
            cameras = cameras_
        else:
            if self.is_animation:
                if not self.showing_overlay:
                    self.buffer_overlay = None
                else:
                    self.update_control_point_overlay()
                fid = torch.tensor(self.animation_time).cuda().float()
            else:
                fid = torch.tensor(0).float().cuda()
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid = fid
            )
            cameras = [cur_cam, cur_cam]
            timesteps = np.array([3] * len(cameras))  # three seconds by default
        
        def min_line_dist_center(rays_o, rays_d):
            try:
                if len(np.shape(rays_d)) == 2:
                    rays_o = rays_o[..., np.newaxis]
                    rays_d = rays_d[..., np.newaxis]
                A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
                b_i = -A_i @ rays_o
                pt_mindist = np.squeeze(-np.linalg.inv((A_i @ A_i).mean(0)) @ (b_i).mean(0))
            except:
                pt_mindist = None
            return pt_mindist

        # Define camera pose keypoints
        vis_cams = []
        c2ws = np.stack([cam.c2w for cam in cameras], axis=0)
        rs = c2ws[:, :3, :3]
        from scipy.spatial.transform import Slerp
        slerp = Slerp(times=np.arange(len(c2ws)), rotations=R.from_matrix(rs))
        from scipy.spatial import geometric_slerp
        
        if rendering_animation:
            from utils.bezier import BezierCurve, PieceWiseLinear
            points = []
            for deform_keypoint_file in deform_keypoint_files:
                if deform_keypoint_file is None:
                    points.append(self.animate_tool.init_pcl.detach().cpu().numpy())
                else:
                    deform_keypoints = load_obj(os.path.join(self.args.model_path, deform_keypoint_file))[0]
                    animated_pcl, _, _ = self.animate_tool.deform_arap(handle_idx=deform_keypoints.get_kpt_idx(), handle_pos=deform_keypoints.get_deformed_kpt_np(), return_R=True)
                    points.append(animated_pcl.detach().cpu().numpy())
            points = np.stack(points, axis=1)
            bezier = PieceWiseLinear(points=points)
        
        # Save path
        sv_dir = os.path.join(self.args.model_path, 'render_trajectory')
        os.makedirs(sv_dir, exist_ok=True)
        import cv2
        video = cv2.VideoWriter(sv_dir + f'/{self.mode}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.W, self.H))

        # Camera loop
        for i in range(len(cameras)-1):
            if use_spiral:
                total_rate = i / (len(cameras) - 1)
                cam = cameras[i]
                if rendering_animation:
                    cam.fid = torch.tensor(self.animation_time).cuda().float()
                    animated_pcl = bezier(t=total_rate)
                    animated_pcl = torch.from_numpy(animated_pcl).cuda()
                    self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                else:
                    cam.fid = torch.tensor(total_rate).cuda().float()
                image = self.test_step(specified_cam=cam)
                image = (image * 255).astype('uint8')
                video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                vis_cams = poses
            else:
                cam0, cam1 = cameras[i], cameras[i+1]
                frame_num = int(timesteps[i] * fps)
                avg_center = min_line_dist_center(c2ws[i:i+2, :3, 3], c2ws[i:i+2, :3, 2])
                if avg_center is not None:
                    vec1_norm1, vec2_norm = np.linalg.norm(c2ws[i, :3, 3] - avg_center), np.linalg.norm(c2ws[i+1, :3, 3] - avg_center)
                    slerp_t = geometric_slerp(start=(c2ws[i, :3, 3]-avg_center)/vec1_norm1, end=(c2ws[i+1, :3, 3]-avg_center)/vec2_norm, t=np.linspace(0, 1, frame_num))
                else:
                    print('avg_center is None. Move along a line.')
                
                for j in range(frame_num):
                    rate = j / frame_num
                    total_rate = (i + rate) / (len(cameras) - 1)
                    if rendering_animation:
                        animated_pcl = bezier(t=total_rate)
                        animated_pcl = torch.from_numpy(animated_pcl).cuda()
                        self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl

                    rot = slerp(i+rate).as_matrix()
                    if avg_center is not None:
                        trans = slerp_t[j] * (vec1_norm1 + (vec2_norm - vec1_norm1) * rate) + avg_center
                    else:
                        trans = c2ws[i, :3, 3] + (c2ws[i+1, :3, 3] - c2ws[i, :3, 3]) * rate
                    c2w = np.eye(4)
                    c2w[:3, :3] = rot
                    c2w[:3, 3] = trans
                    c2w = np.array(c2w, dtype=np.float32)
                    vis_cams.append(c2w)
                    fid = cam0.fid + (cam1.fid - cam0.fid) * rate if not rendering_animation else torch.tensor(self.animation_time).cuda().float()
                    cam = MiniCam(c2w=c2w, width=cam0.image_width, height=cam0.image_height, fovy=cam0.FoVy, fovx=cam0.FoVx, znear=cam0.znear, zfar=cam0.zfar, fid=fid)
                    image = self.test_step(specified_cam=cam)
                    image = (image * 255).astype('uint8')
                    video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()

        print('Trajectory rendered done!')
        self.should_render_customized_trajectory = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(8000, 100_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)

    if args.gui:
        gui.render()
    else:
        gui.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
