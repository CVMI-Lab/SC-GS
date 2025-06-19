import os
import time
import torch
from gaussian_renderer import render
import sys
from scene import GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from scipy.spatial.transform import Rotation as R
import pytorch3d.ops
try:
    from torch_batch_svd import svd
    print('Using speed up torch_batch_svd!')
except:
    svd = torch.svd
    print('Use original torch svd!')


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


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class NodeDriver:
    def __init__(self):
        self.nn_weight = None
        self.nn_dist = None
        self.nn_idxs = None
        self.cached_nn_weight = False
    
    def p2dR(self, p, p0, K=8, as_quat=True):
        p = p.detach()
        nn_dist, nn_idx, nn_nodes = pytorch3d.ops.knn_points(p0[None], p0[None], None, None, K=K+1, return_nn=True)
        nn_dist, nn_idx, nn_nodes = nn_dist[0, :, 1:], nn_idx[0, :, 1:], nn_nodes[0, :, 1:]
        nn_weight = torch.softmax(nn_dist/nn_dist.mean(), dim=-1)
        edges = torch.gather(p0[:, None].expand([p0.shape[0], K, p0.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p0.shape[0], K, p0.shape[-1]])) - p0[:, None]
        t0_deform = None
        edges_t = torch.gather(p[:, None].expand([p.shape[0], K, p.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p.shape[0], K, p.shape[-1]])) - p[:, None]
        edges, edges_t = edges / (edges.norm(dim=-1, keepdim=True) + 1e-5), edges_t / (edges_t.norm(dim=-1, keepdim=True) + 1e-5)
        W = torch.zeros([edges.shape[0], K, K], dtype=torch.float32, device=edges.device)
        W[:, range(K), range(K)] = nn_weight
        S = torch.einsum('nka,nkg,ngb->nab', edges, W, edges_t)
        U, _, V = svd(S)
        dR = torch.matmul(V, U.permute(0, 2, 1))
        if as_quat:
            dR = matrix_to_quaternion(dR)
        return dR, t0_deform

    @torch.no_grad()
    def __call__(self, x, nodes, node_trans_bias, node_radius=1.):
        
        def cal_nn_weight(x:torch.Tensor, nodes, K=None):
            if not self.cached_nn_weight:
                K = self.K if K is None else K
                # Weights of control nodes
                nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K
                nn_weight = torch.exp(- nn_dist / (2 * node_radius ** 2))  # N, K
                nn_weight = nn_weight + 1e-7
                nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
                self.nn_weight = nn_weight
                self.nn_dist = nn_dist
                self.nn_idxs = nn_idxs
                self.cached_nn_weight = True
                return nn_weight, nn_dist, nn_idxs
            else:
                return self.nn_weight, self.nn_dist, self.nn_idxs
        
        x = x.detach()
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        # Animation
        return_dict = {'d_xyz': torch.zeros_like(x), 'd_rotation': 0., 'd_scaling': 0.}
        # Initial nodes and gs
        init_node = nodes
        init_gs = x
        init_nn_weight, _, init_nn_idx = cal_nn_weight(x=init_gs, nodes=init_node, K=8)
        # New nodes and gs
        nodes_t = init_node + node_trans_bias
        node_rot_bias, _ = self.p2dR(p=nodes_t, p0=init_node, K=8, as_quat=True)
        d_nn_node_rot_R = quaternion_to_matrix(node_rot_bias)[init_nn_idx]
        # Aligh the relative distance considering the rotation
        gs_t = nodes_t[init_nn_idx] + torch.einsum('gkab,gkb->gka', d_nn_node_rot_R, (init_gs[:, None] - init_node[init_nn_idx]))
        gs_t_avg = (gs_t * init_nn_weight[..., None]).sum(dim=1)
        translate = gs_t_avg - x
        return_dict['d_xyz'] = translate
        return_dict['d_rotation_bias'] = ((node_rot_bias[init_nn_idx] * init_nn_weight[..., None]).sum(dim=1) - rot_bias) + rot_bias
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        return return_dict


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
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
    def __init__(self, args, pipe) -> None:
        self.args = args
        self.pipe = pipe
        self.gui = True

        self.gaussians = GaussianModel(0)
        self.gaussians.load_ply(args.gs_path)

        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # For UI
        self.visualization_mode = 'RGB'

        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
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
        self.motion_genmodel = None
        self.motion_animation_d_values = None
        self.showing_overlay = True
        self.should_save_screenshot = False
        self.should_vis_trajectory = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')
        self.traj_overlay = None
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

    @torch.no_grad()
    def animation_initialize(self):
        from lap_deform import LapDeform
        pcl = self.gaussians.get_xyz
        from utils.time_utils import farthest_point_sample
        pts_idx = farthest_point_sample(pcl[None], 512)[0]
        pcl = pcl[pts_idx]
        scale = torch.norm(pcl.max(0).values - pcl.min(0).values)
        node_radius = scale / 20
        print(f'Static scene node radius: {node_radius}')
        self.control_nodes = pcl
        self.animate_tool = LapDeform(init_pcl=pcl, K=4, trajectory=None, node_radius=node_radius)
        self.keypoint_idxs = []
        self.keypoint_3ds = []
        self.keypoint_labels = []
        self.keypoint_3ds_delta = []
        self.keypoint_idxs_to_drag = []
        self.deform_keypoints = DeformKeypoints()
        self.animation_trans_bias = None
        self.animation_rot_bias = None
        self.buffer_overlay = None

        self.control_nodes_gaussians = GaussianModel(0)
        from utils.graphics_utils import BasicPointCloud
        colors = self.control_nodes.clone()
        colors = (colors - colors.min(0).values) / (colors.max(0).values - colors.min(0).values)
        pcd = BasicPointCloud(points=pcl.detach().cpu().numpy(), colors=colors.detach().cpu().numpy(), normals=None)
        self.control_nodes_gaussians.create_from_pcd(pcd=pcd)

        self.animator = NodeDriver()
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
        self.animator = NodeDriver()
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
                        # Check the number of edited ply in the model path
                        if not os.path.exists(self.args.model_path):
                            os.makedirs(self.args.model_path)
                        ply_files = sorted([file for file in os.listdir(self.args.model_path) if file.endswith('.ply') and file.startswith('edit')])
                        new_id = len(ply_files)
                        if hasattr(self, 'animator') and self.animation_trans_bias is not None:
                            d_values = self.animator(self.gaussians.get_xyz, self.control_nodes, self.animation_trans_bias)
                            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
                        # Deep copy a new Gaussian
                        import copy
                        gaussian_new = copy.deepcopy(self.gaussians)
                        gaussian_new._xyz = gaussian_new.get_xyz + d_xyz
                        gaussian_new._rotation = gaussian_new.get_rotation * d_rotation
                        gaussian_new.save_ply('{}/edit_{}.ply'.format(self.args.model_path, new_id))
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
                    

            # Saving stuff
            with dpg.collapsing_header(label="Save", default_open=True):
                with dpg.group(horizontal=True):
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
                            dpg.configure_item("_button_overlay", label="show overlay")
                        else:
                            self.showing_overlay = True
                            dpg.configure_item("_button_overlay", label="close overlay")                    
                    dpg.add_button(
                        label="close overlay", tag="_button_overlay", callback=callback_overlay
                    )
                    dpg.bind_item_theme("_button_overlay", theme_button)

                    def callback_save_ckpt(sender, app_data):
                        from utils.pickle_utils import save_obj
                        if not self.is_animation:
                            print('Switch to animation mode!')
                            self.is_animation = True
                            self.animation_initialize()
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
            if dpg.is_key_down(dpg.mvKey_R) or dpg.is_key_down(dpg.mvKey_Q):
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
                    print("Switch to animation mode!")
                    self.is_animation = True
                    self.animation_initialize()
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
                    out = render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=0, d_rotation=0, d_scaling=0)

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])

                    d = out['depth'][0][ph, pw]
                    z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * d - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                    uvz = torch.tensor([((pw-.5)/self.W * 2 - 1) * d, ((ph-.5)/self.H*2-1) * d, z, d]).cuda().float().view(1, 4)
                    p3d = (uvz @ torch.inverse(cur_cam.full_proj_transform))[0, :3]

                    # Pick the closest node as the keypoint
                    nodes = self.control_nodes + self.animation_trans_bias if self.animation_trans_bias is not None else self.control_nodes
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
   
    # gui mode
    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            if self.should_render_customized_trajectory:
                self.render_customized_trajectory(use_spiral=self.should_render_customized_trajectory_spiral)
            self.test_step()

            dpg.render_dearpygui_frame()
    
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
                0
            )
        fid = cur_cam.fid

        vis_scale_const = None
        d_rotation_bias = None
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
        if 'Node' in self.visualization_mode:
            d_xyz = self.animation_trans_bias if self.animation_trans_bias is not None else 0.
            gaussians = self.control_nodes_gaussians
        else:
            if hasattr(self, 'animator') and self.animation_trans_bias is not None:
                d_values = self.animator(self.gaussians.get_xyz, self.control_nodes, self.animation_trans_bias)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            gaussians = self.gaussians
        
        out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, scale_const=vis_scale_const, d_rotation_bias=d_rotation_bias)

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

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)}")
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
                mv[0, 3] *= -1
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
    
    def render_customized_trajectory(self, use_spiral=False, traj_dir=None, fps=30, motion_repeat=1):
        from utils.pickle_utils import load_obj
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
            c2ws = [self.cam.pose.copy()]
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
    pp = PipelineParams(parser)
    
    parser.add_argument('--gs_path', type=str, required=True, help="path to the Gaussian Splatting model")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--white_background', action='store_true', default=False, help="use white background in GUI")
    parser.add_argument('--model_path', type=str, default='./', help="path to save the model and logs")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(8000, 100_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, pipe=pp.extract(args))

    gui.render()
    
    # All done
    print("\nTraining complete.")
