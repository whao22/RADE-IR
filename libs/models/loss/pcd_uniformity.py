import torch
import pytorch3d.ops as ops

def get_pcd_uniformity_loss(vertices: torch.Tensor):
    """The function calculates the pcd uniformality loss of a point cloud, which is used 
    to regularize the shape of the point cloud.

    Args:
        vertices (torch.Tensor): [1, V, 3]. The input point cloud.

    Returns:
        _type_: _description_
    """
    knn_ret = ops.knn_points(vertices, vertices.clone(), K=K+1)
    p_idx, p_squared_dist = knn_ret.idx, knn_ret.dists
    p_dist = p_squared_dist[..., 1:] ** 0.5 # [1, V, K]
    nearest_dist = p_dist.mean(dim=-1) # [1, V]
    mean_dist = torch.mean(nearest_dist, dim=-1, keepdim=True)
    pcd_uniformality_loss = torch.nn.functional.l1_loss(nearest_dist, mean_dist.repeat(1, nearest_dist.shape[1]))
    
    return pcd_uniformality_loss