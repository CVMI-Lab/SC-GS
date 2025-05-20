import numpy as np
import torch
from pytorch3d.loss.mesh_laplacian_smoothing import cot_laplacian
from pytorch3d.ops import ball_query
from pytorch3d.io import load_ply
# try:
#     print('Using speed up torch_batch_svd!')
#     from torch_batch_svd import svd
# except:
#     print('Use original torch svd!')
svd = torch.svd
import pytorch3d.ops


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


def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda") -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a tensor E, where, for neighbour list J,
	E_in = p_i - p_(J[n])"""

	E = torch.zeros(edge_shape).to(device)
	E[ii, nn] = verts[ii] - verts[jj]

	return E


####################### utils for arap #######################

def geodesic_distance_floyd(cur_node, K=8):
    node_num = cur_node.shape[0]
    nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(cur_node[None], cur_node[None], None, None, K=K+1)
    nn_dist, nn_idx = nn_dist[0]**.5, nn_idx[0]
    dist_mat = torch.inf * torch.ones([node_num, node_num], dtype=torch.float32, device=cur_node.device)
    dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
    dist_mat = torch.minimum(dist_mat, dist_mat.T)
    for i in range(nn_dist.shape[0]):
        dist_mat = torch.minimum((dist_mat[:, i, None] + dist_mat[None, i, :]), dist_mat)
    return dist_mat

def cal_connectivity_from_points(points=None, radius=0.1, K=10, trajectory=None, least_edge_num=3, node_radius=None, mode='nn', GraphK=4, adaptive_weighting=True):
     # input: [Nv,3]
     # output: information of edges
     # ii : [Ne,] the i th vert
     # jj: [Ne,] j th vert is connect to i th vert.
     # nn: ,  [Ne,] the n th neighbour of i th vert is j th vert.
    Nv = points.shape[0] if points is not None else trajectory.shape[0]
    if trajectory is None:
        if mode == 'floyd':
            dist_mat = geodesic_distance_floyd(points, K=GraphK)
            dist_mat = dist_mat ** 2
            mask = torch.eye(Nv).bool()
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            knn_res = pytorch3d.ops.knn_points(points[None], points[None], None, None, K=K+1)
            # Remove themselves
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K], [Nv, K]
    else:
        trajectory = trajectory.reshape([Nv, -1]) / trajectory.shape[1]  # Average distance of trajectory
        if mode == 'floyd':
            dist_mat = geodesic_distance_floyd(trajectory, K=GraphK)
            dist_mat = dist_mat ** 2
            mask = torch.eye(Nv).bool()
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            knn_res = pytorch3d.ops.knn_points(trajectory[None], trajectory[None], None, None, K=K+1)
            # Remove themselves
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K], [Nv, K]

    # Make sure ranges are within the radius
    nn_idx[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_idx[:, least_edge_num:], - torch.ones_like(nn_idx[:, least_edge_num:]))
    
    nn_dist[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_dist[:, least_edge_num:], torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf)
    if adaptive_weighting:
        nn_dist_1d = nn_dist.reshape(-1)
        weight = torch.exp(-nn_dist / nn_dist_1d[~torch.isnan(nn_dist_1d) & ~torch.isinf(nn_dist_1d)].mean())
    elif node_radius is None:
        weight = torch.exp(-nn_dist)
    else:
        nn_radius = node_radius[nn_idx]
        weight = torch.exp(-nn_dist / (2 * nn_radius ** 2))
    weight = weight / weight.sum(dim=-1, keepdim=True)

    ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K).reshape([-1])
    jj = nn_idx.reshape([-1])
    nn = torch.arange(K)[None].cuda().long().expand(Nv, K).reshape([-1])
    mask = jj != -1
    ii, jj, nn = ii[mask], jj[mask], nn[mask]

    return ii, jj, nn, weight


def cal_laplacian(Nv, ii, jj, nn):
    # input: Nv: int; ii, jj, nn: [Ne,]
    # output: laplacian_mat: [Nv, Nv]
    laplacian_mat = torch.zeros(Nv, Nv).cuda()
    laplacian_mat[ii, jj] = -1
    for idx in ii:
        laplacian_mat[idx, idx] += 1  # TODO test whether it is correct
    return laplacian_mat

def cal_verts_deg(Nv, ii):
    # input: Nv: int; ii, jj, nn: [Ne,]
    # output: verts_deg: [Nv,]
    verts_deg = torch.zeros(Nv).cuda()
    for idx in ii:
        verts_deg[idx] += 1
    return verts_deg

def estimate_rotation(source, target, ii, jj, nn, K=10, weight=None, sample_idx=None):
    # input: source, target: [Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: rotation: [Nv, 3, 3]
    Nv = len(source)
    source_edge_mat = produce_edge_matrix_nfmt(source, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    target_edge_mat = produce_edge_matrix_nfmt(target, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
        print("!!! Edge weight is None !!!")
    if sample_idx is not None:
        source_edge_mat = source_edge_mat[sample_idx]
        target_edge_mat = target_edge_mat[sample_idx]
    ### Calculate covariance matrix in bulk
    D = torch.diag_embed(weight, dim1=1, dim2=2)  # [Nv, K, K]
    # S = torch.bmm(source_edge_mat.permute(0, 2, 1), target_edge_mat)  # [Nv, 3, 3]
    S = torch.bmm(source_edge_mat.permute(0, 2, 1), torch.bmm(D, target_edge_mat))  # [Nv, 3, 3]
    ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
    unchanged_verts = torch.unique(torch.where((source_edge_mat == target_edge_mat).all(dim=1))[0])  # any verts which are undeformed
    S[unchanged_verts] = 0
    
    # t2 = time.time()
    U, sig, W = svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations
    # t3 = time.time()

    # Need to flip the column of U corresponding to smallest singular value
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
        Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
        R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))
    # t4 = time.time()
    # print(f'0-1: {t1-t0}, 1-2: {t2-t1}, 2-3: {t3-t2}, 3-4: {t4-t3}')
    return R

def invert_matrix(mat):
    try:
        mat_inv = torch.inverse(mat)
    except:
        print("L_reduced is not invertible, use pseudo inverse instead")
        mat_inv = torch.linalg.pinv(mat)
    return mat_inv

import time
def cal_arap_error(nodes_sequence, ii, jj, nn, K=10, weight=None, sample_num=512):
    # input: nodes_sequence: [Nt, Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: arap error: float
    Nt, Nv, _ = nodes_sequence.shape
    # laplacian_mat = cal_laplacian(Nv, ii, jj, nn)  # [Nv, Nv]
    # laplacian_mat_inv = invert_matrix(laplacian_mat)
    arap_error = 0
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
    source_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[0], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]
    for idx in range(1, Nt):
        # t1 = time.time()
        with torch.no_grad():
            rotation = estimate_rotation(nodes_sequence[0], nodes_sequence[idx], ii, jj, nn, K=K, weight=weight, sample_idx=sample_idx)  # [Nv, 3, 3]
        # Compute energy
        target_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[idx], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
        target_edge_mat = target_edge_mat[sample_idx]
        rot_rigid = torch.bmm(rotation, source_edge_mat[sample_idx].permute(0, 2, 1)).permute(0, 2, 1)  # [Nv, K, 3]
        stretch_vec = target_edge_mat - rot_rigid  # stretch vector
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
        arap_error += (weight * stretch_norm).sum()
    return arap_error

def cal_L_from_points(points, return_nn_idx=False):
    # points: (N, 3)
    Nv = len(points)
    L = torch.eye(Nv).cuda()
    radius = 0.1  # 
    K = 20
    knn_res = ball_query(points[None], points[None], K=K, radius=radius, return_nn=False)
    nn_dist, nn_idx = knn_res.dists[0], knn_res.idx[0]  # [Nv, K], [Nv, K]
    for idx, cur_nn_idx in enumerate(nn_idx):
        real_cur_nn_idx = cur_nn_idx[cur_nn_idx != -1]
        real_cur_nn_idx = real_cur_nn_idx[real_cur_nn_idx != idx]
        L[idx, idx] = len(real_cur_nn_idx)
        L[idx][real_cur_nn_idx] = -1
    if return_nn_idx:
        return L, nn_idx
    else:
        return L

def lstsq_with_handles(A, b, handle_idx, handle_pos, A_is_degenarate=False):
    b = b - A[:, handle_idx] @ handle_pos
    handle_mask = torch.zeros_like(A[:, 0], dtype=bool)
    handle_mask[handle_idx] = 1
    L = A[:, handle_mask.logical_not()]
    if not A_is_degenarate:
        x = torch.linalg.lstsq(L, b)[0]
    else:
        x = torch.linalg.pinv(L) @ b
    x_out = torch.zeros_like(b)
    x_out[handle_idx] = handle_pos
    x_out[handle_mask.logical_not()] = x
    return x_out

def rigid_align(x, y):
    x_bar, y_bar = x.mean(0), y.mean(0)
    x, y = x - x_bar, y - y_bar
    S = x.permute(1, 0) @ y  # 3 * 3
    U, _, W = svd(S)
    R = W @ U.permute(1, 0)
    t = y_bar - R @ x_bar
    x2y = x @ R.T + t
    return x2y, R, t

def arap_deformation_loss(trajectory, node_radius=None, trajectory_rot=None, K=50, with_rot=True):
    init_pcl = trajectory[:, 0]
    radius = torch.linalg.norm(init_pcl.max(dim=0).values - init_pcl.min(dim=0).values) / 8
    fid = torch.randint(1, trajectory.shape[1], [])
    tar_pcl = trajectory[:, fid]

    N = init_pcl.shape[0]
    with torch.no_grad():
        radius = torch.linalg.norm(init_pcl.max(dim=0).values - init_pcl.min(dim=0).values) / 8
        device = init_pcl.device
        ii, jj, nn, weight = cal_connectivity_from_points(init_pcl, radius, K, trajectory=trajectory.detach(), node_radius=node_radius, mode='nn')
        L_opt = torch.eye(N).cuda()
        L_opt[ii, jj] = - weight[ii, nn]
    
    P = produce_edge_matrix_nfmt(init_pcl, (N, K, 3), ii, jj, nn, device=device)
    P_prime = produce_edge_matrix_nfmt(tar_pcl, (N, K, 3), ii, jj, nn, device=device)
    
    with torch.no_grad():
        D = torch.diag_embed(weight, dim1=1, dim2=2)
        S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))
        U, sig, W = torch.svd(S)
        R = torch.bmm(W, U.permute(0, 2, 1))
        with torch.no_grad():
            # Need to flip the column of U corresponding to smallest singular value
            # for any det(Ri) <= 0
            entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
            if len(entries_to_flip) > 0:
                Umod = U.clone()
                cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
                Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
                R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))
    arap_error = (weight[..., None] * (P_prime - torch.einsum('bxy,bky->bkx', R, P))).square().mean(dim=0).sum()

    if with_rot:
        init_rot = quaternion_to_matrix(trajectory_rot[:, 0])
        tar_rot = quaternion_to_matrix(trajectory_rot[:, fid])
        R_rot = torch.bmm(R, init_rot)
        rot_error = (R_rot - tar_rot).square().mean(dim=0).sum()
    else:
        rot_error = 0.

    return arap_error, rot_error * 1e2
