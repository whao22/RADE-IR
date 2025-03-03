import torch

def compute_normal_consistency(deformed_gaussian, normals: torch.Tensor, k_neighbors=3):
    """
    Compute normal consistency loss for a point cloud.
    The loss is computed by finding the k nearest neighbors of each point and calculating the cosine similarity between
    the point's normal and its neighbors' normals. The loss is the average of the cosine similarities.
    Args:
        normals: [N, 3], normal vector of each point
        k_neighbors: int, number of nearest neighbors to consider
    Returns:
        consistency_loss: torch.Tensor, normal consistency loss
    """
    xyz = deformed_gaussian.get_xyz.detach()
    
    # find k nearest neighbors and get neighbors' normals
    dist = torch.cdist(xyz, xyz)
    _, indices = torch.topk(dist, k=k_neighbors+1, largest=False)
    indices = indices[:, 1:]
    neighbor_normals = normals[indices]

    # calculate the cosine similarity between each point's normal and its neighbors' normals
    normals = normals.unsqueeze(1).repeat([1, k_neighbors, 1]).view(-1, 3) # [M, 3]
    neighbor_normals = neighbor_normals.view(-1, 3) # [M, 3]
    dot_products = (normals.unsqueeze(1) @ neighbor_normals.unsqueeze(-1)).unsqueeze(1) # [M, 1]
    normal_consistency = 1 - torch.mean(dot_products)

    return normal_consistency