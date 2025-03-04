import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import numpy as np
from omegaconf import OmegaConf

from libs.utils.loss_utils import l1_loss, ssim, get_pcd_uniformity_loss, l1_loss_appearance, get_masked_tv_loss, first_order_edge_aware_loss
from libs.utils.graphics_utils import point_double_to_normal, depth_double_to_normal
from libs.utils.loss_utils import full_aiap_loss

def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


def compute_loss(iteration, config, dataset, data, render_pkg, scene, loss_fn_vgg, reg_kick_on, require_depth):
    loss_dict = {}
    deformed_gaussian = render_pkg["deformed_gaussian"]
    
    # render_image loss
    image =  render_pkg["rendered_image"]
    gt_image = data.original_image.cuda()
    
    lambda_l1 = C(iteration, config.opt.lambda_l1)
    lambda_dssim = C(iteration, config.opt.lambda_dssim)
    loss_l1 = torch.tensor(0.).cuda()
    loss_dssim = torch.tensor(0.).cuda()
    if lambda_l1 > 0.:
        if dataset.use_decoupled_appearance:
            loss_l1 = l1_loss_appearance(image, gt_image, deformed_gaussian, data.frame_id)
        else:
            loss_l1 = l1_loss(image, gt_image)
    if lambda_dssim > 0.:
        loss_dssim = 1.0 - ssim(image, gt_image)
    loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim
    loss_dict.update({
        "loss_l1": loss_l1,
        "loss_dssim": loss_dssim,
    })
    
    # rendered_image2 loss
    image2 =  render_pkg["rendered_image2"]
    loss_l1_2 = torch.tensor(0.).cuda()
    loss_dssim_2 = torch.tensor(0.).cuda()
    if lambda_l1 > 0.:
        if dataset.use_decoupled_appearance:
            loss_l1_2 = l1_loss_appearance(image2, gt_image, deformed_gaussian, data.frame_id)
        else:
            loss_l1_2 = l1_loss(image, gt_image)
    if lambda_dssim > 0.:
        loss_dssim_2 = 1.0 - ssim(image, gt_image)
    loss += lambda_l1 * loss_l1_2 + lambda_dssim * loss_dssim_2
    loss_dict.update({
        "loss_l1_2": loss_l1_2,
        "loss_dssim_2": loss_dssim_2,
    })

    # perceptual loss
    lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
    if lambda_perceptual > 0:
        # crop the foreground
        mask = data.original_mask.cpu().numpy()
        mask = np.where(mask)
        y1, y2 = mask[1].min(), mask[1].max() + 1
        x1, x2 = mask[2].min(), mask[2].max() + 1
        fg_image = image[:, y1:y2, x1:x2]
        gt_fg_image = gt_image[:, y1:y2, x1:x2]

        loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
        loss += lambda_perceptual * loss_perceptual
    else:
        loss_perceptual = torch.tensor(0.)
    loss_dict["loss_perceptual"] = loss_perceptual

    # mask loss
    lambda_mask = C(iteration, config.opt.lambda_mask)
    gt_mask = data.original_mask.cuda()
    opacity = render_pkg["opacity_render"]
    if lambda_mask <= 0:
        loss_mask = torch.tensor(0.).cuda()
    elif config.opt.mask_loss_type == 'bce':
        opacity = torch.clamp(opacity, 1.e-3, 1.-1.e-3)
        loss_mask = F.binary_cross_entropy(opacity, gt_mask)
    elif config.opt.mask_loss_type == 'l1':
        loss_mask = F.l1_loss(opacity, gt_mask)
    else:
        raise ValueError
    loss += lambda_mask * loss_mask
    loss_dict["loss_mask"] = loss_mask

    # skinning loss
    lambda_skinning = C(iteration, config.opt.lambda_skinning)
    if lambda_skinning > 0:
        loss_skinning = scene.get_skinning_loss()
        loss += lambda_skinning * loss_skinning
    else:
        loss_skinning = torch.tensor(0.).cuda()

    lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
    lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
    if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
        loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
    else:
        loss_aiap_xyz = torch.tensor(0.).cuda()
        loss_aiap_cov = torch.tensor(0.).cuda()
    loss += lambda_aiap_xyz * loss_aiap_xyz
    loss += lambda_aiap_cov * loss_aiap_cov
    loss_dict["loss_aiap_xyz"] = loss_aiap_xyz
    loss_dict["loss_aiap_cov"] = loss_aiap_cov

    # pcd uniformity loss
    lambda_pcd_uniformity = C(iteration, config.opt.lambda_pcd_uniformity)
    if lambda_pcd_uniformity > 0:
        loss_pcd_uniformity = get_pcd_uniformity_loss(deformed_gaussian.get_xyz.unsqueeze(0), K=5)
        loss += lambda_pcd_uniformity * loss_pcd_uniformity
    else:
        loss_pcd_uniformity = torch.tensor(0.).cuda()
    loss_dict["loss_pcd_uniformity"] = loss_pcd_uniformity

    # geometry regularization
    if reg_kick_on:
        lambda_depth_normal = config.opt.lambda_depth_normal
        if require_depth:
            rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
            rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
            rendered_normal: torch.Tensor = render_pkg["rendered_normal"]
            depth_middepth_normal = depth_double_to_normal(data, rendered_expected_depth, rendered_median_depth)
        else:
            rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
            rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
            rendered_normal: torch.Tensor = render_pkg["rendered_normal"]
            depth_middepth_normal = point_double_to_normal(data, rendered_expected_coord, rendered_median_coord)
        depth_ratio = 0.6
        normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
        loss_depth_normal = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
    else:
        lambda_depth_normal = 0
        loss_depth_normal = torch.tensor(0.).cuda()
    loss += lambda_depth_normal * loss_depth_normal
    loss_dict["loss_depth_normal"] = loss_depth_normal

    ################# normal #####################
    # normal image
    rendered_normal2 = render_pkg["rendered_normal2"]
    rendered_normal = render_pkg["rendered_normal"].detach()
    if lambda_l1 > 0.:
        loss_normal_l1 = l1_loss(rendered_normal2, rendered_normal)
    else:
        loss_normal_l1 = torch.tensor(0).cuda()
    if lambda_dssim > 0.:
        loss_normal_dssim = 1.0 - ssim(rendered_normal2, rendered_normal)
    else:
        loss_normal_dssim = torch.tensor(0).cuda()
    loss += lambda_l1 * loss_normal_l1 + lambda_dssim * loss_normal_dssim
    loss_dict.update({
        "loss_normal_l1": loss_normal_l1,
        "loss_normal_dssim": loss_normal_dssim,
    })
    ################# normal #####################
    
    # pbr loss
    lambda_pbr = C(iteration, config.opt.lambda_pbr)
    if lambda_pbr > 0:
        rendered_pbr = render_pkg["rendered_pbr"]
        if lambda_l1 > 0.:
            l1_pbr = l1_loss(rendered_pbr, gt_image)
        else:
            l1_pbr = torch.tensor(0.).cuda()
        if lambda_dssim > 0.:
            ldssim_pbr = 1.0 - ssim(rendered_pbr, gt_image)
        else:
            ldssim_pbr = torch.tensor(0.).cuda()
        loss_pbr = lambda_l1 * l1_pbr + lambda_dssim * ldssim_pbr
        loss += lambda_pbr * loss_pbr
    else:
        loss_pbr = torch.tensor(0.).cuda()
    loss_dict["loss_pbr"] = loss_pbr

    # base color loss
    lambda_base_color_smooth = C(iteration, config.opt.lambda_base_color_smooth)
    if lambda_base_color_smooth > 0:
        loss_base_color = first_order_edge_aware_loss(render_pkg["albedo_map"] * gt_mask, gt_image)
        loss += lambda_base_color_smooth * loss_base_color
    else:
        loss_base_color = torch.tensor(0.).cuda()
    loss_dict["loss_base_color"] = loss_base_color

    # roughtness loss
    lambda_metallic_smooth = C(iteration, config.opt.lambda_metallic_smooth)
    if lambda_metallic_smooth > 0:
        loss_roughness= first_order_edge_aware_loss(render_pkg["roughness_map"] * gt_mask, gt_image)
        loss += lambda_metallic_smooth * loss_roughness
    else:
        loss_roughness = torch.tensor(0.).cuda()
    loss_dict["loss_roughness"] = loss_roughness

    # metallic loss
    lambda_metallic_smooth = C(iteration, config.opt.lambda_metallic_smooth)
    if lambda_metallic_smooth > 0:
        loss_metallic = first_order_edge_aware_loss(render_pkg["metallic_map"] * gt_mask, gt_image)
        loss += lambda_metallic_smooth * loss_metallic
    else:
        loss_metallic = torch.tensor(0.).cuda()
    loss_dict["loss_metallic"] = loss_metallic

    # brdf tv loss
    lambda_brdf_tv = C(iteration, config.opt.lambda_brdf_tv)
    if lambda_brdf_tv > 0:
        albedo_map = render_pkg["albedo_map"]
        roughness_map = render_pkg["roughness_map"]
        metallic_map = render_pkg["metallic_map"]
        
        loss_brdf_tv = get_masked_tv_loss(
            gt_mask,
            gt_image,  # [3, H, W]
            torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
        )
        loss += lambda_brdf_tv * loss_brdf_tv
    else:
        loss_brdf_tv = torch.tensor(0.).cuda()
    loss_dict["loss_brdf_tv"] = loss_brdf_tv

    # envmap tv smoothness
    lambda_env_tv = C(iteration, config.opt.lambda_env_tv)
    if lambda_env_tv > 0:
        envmap = dr.texture(
            scene.cubemap.base[None, ...],
            scene.envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[0]  # [H, W, 3]
        tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
        tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
        loss_env_tv = tv_h1 + tv_w1
        loss += loss_env_tv * lambda_env_tv
    else:
        loss_env_tv = torch.tensor(0.).cuda()
    loss_dict["loss_env_tv"] = loss_env_tv

    # regularization
    loss_reg = render_pkg["loss_reg"]
    for name, value in loss_reg.items():
        lbd = config.opt.get(f"lambda_{name}", 0.)
        lbd = C(iteration, lbd)
        loss += lbd * value
        loss_dict[f"loss_{name}"] = value

    return loss, loss_dict