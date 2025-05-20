import numpy as np
import torch
from utils.deform_utils import cal_laplacian, cal_connectivity_from_points,\
      produce_edge_matrix_nfmt, lstsq_with_handles, cal_verts_deg, rigid_align
from utils.other_utils import matrix_to_quaternion


def cal_L_from_points(points, return_nn_idx=False):
    # points: (N, 3)
    Nv = len(points)
    L = torch.eye(Nv).cuda()

    radius = 0.3  # 
    K = 10
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
    

def mask_softmax(x, mask, dim=1):
    # x: (N, K), mask: (N, K) 0/1
    x = torch.exp(x)
    x = x * mask
    x = x / x.sum(dim=dim, keepdim=True)
    return x


class ARAPDeformer:
    def __init__(self, verts, K=10, radius=0.3, point_mask=None, trajectory=None, node_radius=None) -> None:
        # verts: (N, 3), one_ring_idx: (N, K)
        self.device = verts.device
        self.verts = verts
        self.verts_copy = verts.clone()
        self.radius = radius
        self.K = K
        self.N = len(verts)

        self.ii, self.jj, self.nn, weight = cal_connectivity_from_points(self.verts, self.radius, self.K, trajectory=trajectory, node_radius=node_radius)
        self.L = cal_laplacian(Nv=self.N, ii=self.ii, jj=self.jj, nn=self.nn)
        # self.L = cal_L_from_points(points=self.verts)

        ##### add learnable deformation weights #####
        self.vert_deg = cal_verts_deg(self.N, self.ii)
        # weight = torch.ones(self.N, K).float().cuda()  # [Nv, K]
        # weight[self.ii, self.nn] = -1 / self.vert_deg[self.ii]
        self.weight = torch.nn.Parameter(weight, requires_grad=True)  # [Nv, K]
        self.weight_mask = torch.zeros(self.N, K).float().cuda()  # [Nv, K]
        self.weight_mask[self.ii, self.nn] = 1

        self.L_opt = torch.eye(self.N).cuda()  # replace all the self.L with self.L_opt! s.t. weight is in [0,1], easy to optimize.
        self.L_is_degenerate = False
        self.cal_L_opt()
        self.b = torch.mm(self.L_opt, self.verts)  # [Nv, 3]

        self.point_mask = point_mask  # [N,]

    def cal_L_opt(self):
        self.normalized_weight = self.weight
        self.L_opt[self.ii, self.jj] = - self.normalized_weight[self.ii, self.nn]  # [Nv, Nv]
        self.L_is_degenerate = (torch.linalg.matrix_rank(self.L_opt) < self.L_opt.shape[0]) 
        if self.L_is_degenerate:
            print("L_opt is not invertible, use pseudo inverse instead")

    def reset(self):
        self.verts = self.verts_copy.clone()

    def world_2_local_index(self, handle_idx):
        # handle_idx: [m,]
        # point mask [N,]
        # idx_offset = torch.cat([torch.zeros_like(self.point_mask[:1]), torch.cumsum(self.point_mask, dim=0)])
        idx_offset = torch.cumsum(~self.point_mask, dim=0)
        handle_idx_offset = idx_offset[handle_idx]
        return handle_idx - handle_idx_offset


    def deform(self, handle_idx, handle_pos, init_verts=None, return_R=False):
        # handle_idx: (M, ), handle_pos: (M, 3)

        if self.point_mask is not None:
            handle_idx = self.world_2_local_index(handle_idx)

        ##### calculate b #####
        ### b_fixed
        unknown_verts = [n for n in range(self.N) if n not in handle_idx.tolist()]  # all unknown verts
        b_fixed = torch.zeros((self.N, 3), device=self.device)  # factor to be subtracted from b, due to constraints
        for k, pos in zip(handle_idx, handle_pos):
            # b_fixed += torch.einsum("i,j->ij", self.L[:, k], pos)  # [Nv,3]
            b_fixed += torch.einsum("i,j->ij", self.L_opt[:, k], pos)  # [Nv,3]
        
        ### prepare for b_all
        P = produce_edge_matrix_nfmt(self.verts, (self.N, self.K, 3), self.ii, self.jj, self.nn, device=self.device)  # [Nv, K, 3]
        if init_verts is None:
            p_prime = lstsq_with_handles(self.L_opt, self.L_opt@self.verts, handle_idx, handle_pos, A_is_degenarate=self.L_is_degenerate)
        else:
            p_prime = init_verts
        
        p_prime_seq = [p_prime]
        R = torch.eye(3)[None].repeat(self.N, 1,1).cuda()  # compute rotations
        
        NUM_ITER = 3
        D = torch.diag_embed(self.normalized_weight, dim1=1, dim2=2)  # [Nv, K, K]
        for _ in range(NUM_ITER):
            P_prime = produce_edge_matrix_nfmt(p_prime, (self.N, self.K, 3), self.ii, self.jj, self.nn, device=self.device)  # [Nv, K, 3]
            ### Calculate covariance matrix in bulk
            S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))  # [Nv, 3, 3]

            ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
            unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0])  # any verts which are undeformed
            S[unchanged_verts] = 0

            U, sig, W = torch.svd(S)
            R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

            # Need to flip the column of U corresponding to smallest singular value
            # for any det(Ri) <= 0
            entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
            if len(entries_to_flip) > 0:
                Umod = U.clone()
                cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
                Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
                R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

            ### RHS of minimum energy equation
            Rsum_shape = (self.N, self.K, 3, 3)
            Rsum = torch.zeros(Rsum_shape).to(self.device)  # Ri + Rj, as in eq (8)
            Rsum[self.ii, self.nn] = R[self.ii] + R[self.jj]
            
            ### Rsum has shape (V, max_neighbours, 3, 3). P has shape (V, max_neighbours, 3)
            ### To batch multiply, collapse first 2 dims into a single batch dim
            Rsum_batch, P_batch = Rsum.view(-1, 3, 3), P.view(-1, 3).unsqueeze(-1)
            
            # RHS of minimum energy equation
            b = 0.5 * (torch.bmm(Rsum_batch, P_batch).squeeze(-1).reshape(self.N, self.K, 3) * self.normalized_weight[...,None]).sum(dim=1)

            ### calculate p_prime
            p_prime = lstsq_with_handles(self.L_opt, b, handle_idx, handle_pos, A_is_degenarate=self.L_is_degenerate)

            p_prime_seq.append(p_prime)
        d_scaling = None

        if return_R:
            quat = matrix_to_quaternion(R)
            return p_prime, quat, d_scaling
        else:
            # return p_prime, p_prime_seq
            return p_prime
    


if __name__ == "__main__":
    from pytorch3d.io import load_ply
    from pytorch3d.ops import ball_query
    import pickle
    with open("./control_kpt.pkl", "rb") as f:
        data = pickle.load(f)

    points = data["pts"]
    handle_idx = data["handle_idx"]
    handle_pos = data["handle_pos"]

    import trimesh
    trimesh.Trimesh(vertices=points).export('deformation_before.ply')

    #### prepare data
    points = torch.from_numpy(points).float().cuda()
    handle_idx = torch.tensor(handle_idx).long().cuda()
    handle_pos = torch.from_numpy(handle_pos).float().cuda()

    deformer = ARAPDeformer(points)

    with torch.no_grad():
        points_prime, p_prime_seq = deformer.deform(handle_idx, handle_pos)

    trimesh.Trimesh(vertices=points_prime.cpu().numpy()).export('deformation_after.ply')

    from utils.deform_utils import cal_arap_error
    for p_prime in p_prime_seq:
        nodes_sequence = torch.cat([points[None], p_prime[None]], dim=0)
        arap_error = cal_arap_error(nodes_sequence, deformer.ii, deformer.jj, deformer.nn, K=deformer.K, weight=deformer.normalized_weight)
        print(arap_error)