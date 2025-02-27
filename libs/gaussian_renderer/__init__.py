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
import torch
import torch.nn.functional as F
import math

from libs.utils.pbr import pbr_shading
from libs.scene import Scene
from radegs_rasterization import GaussianRasterizationSettings as DeRasterSettings, GaussianRasterizer as DeRasterizer
from r3dg_rasterization import GaussianRasterizationSettings as R3DRasterSettings, GaussianRasterizer as R3DRasterizer

def pbr_render(data, scene: Scene, rendered_maps: list, bg_color):
    rendered_base_color, rendered_metallic, rendered_roughness, rendered_normal, rendered_alpha, rendered_median_depth, rendered_visibility = rendered_maps
    
    # formulate roughness
    rmax, rmin = 1.0, 0.04
    rendered_roughness = rendered_roughness * (rmax - rmin) + rmin

    # PBR rendering
    rays = scene.get_canonical_rays(data)
    c2w = torch.inverse(data.world_view_transform.T)  # [4, 4]
    view_dirs = -(
        (F.normalize(rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
        .sum(dim=-1)
        .reshape(data.image_height, data.image_width, 3)
    )  # [H, W, 3]
    points = (
        (-view_dirs.reshape(-1, 3) * rendered_median_depth.reshape(-1, 1) + c2w[:3, 3])
        .clamp(min=-scene.cfg.opt.bound, max=scene.cfg.opt.bound)
        .contiguous()
    )  # [HW, 3]
    
    irradiance_map = scene.irradiance_volumes.query_irradiance(
        points=points.reshape(-1, 3).contiguous(),
        normals=rendered_normal.permute(1, 2, 0).reshape(-1, 3).contiguous(),
    ).reshape(data.image_height, data.image_width, -1)
    
    scene.cubemap.build_mips() # build mip for environment light
    normal_mask = (rendered_alpha != 0).all(0, keepdim=True)
    pbr_result = pbr_shading(
        view_dirs=view_dirs,
        light=scene.cubemap,
        brdf_lut=scene.brdf_lut,
        normals=rendered_normal.permute(1, 2, 0).detach(),  # [H, W, 3]
        mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
        albedo=rendered_base_color.permute(1, 2, 0),  # [H, W, 3]
        roughness=rendered_roughness.permute(1, 2, 0),  # [H, W, 1]
        metallic=rendered_metallic.permute(1, 2, 0),  # [H, W, 1]
        occlusion=rendered_visibility.permute(1, 2, 0),  # [H, W, 1]
        irradiance=irradiance_map,
        tone=scene.cfg.opt.tone,
        gamma=scene.cfg.opt.gamma,
        background=bg_color,
    )
    
    return pbr_result


def render(data,
           iteration,
           scene: Scene,
           pipe,
           bg_color : torch.Tensor,
           kernel_size : int,
           scaling_modifier = 1.0,
           require_coord : bool = True,
           compute_loss=True,
           require_depth=True,
           is_training=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp, intrinsic_precompute = scene.convert_gaussians(data, iteration, compute_loss)
    
    results = {
        "deformed_gaussian": pc,
    }
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    rade_raster_settings = DeRasterSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=pipe.debug
    )
    r3d_raster_settings = R3DRasterSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(data.K[0, 2]),
        cy=float(data.K[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rade_rasterizer = DeRasterizer(raster_settings=rade_raster_settings)
    r3d_rasterizer = R3DRasterizer(raster_settings=r3d_raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    _scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    
    if pipe.compute_cov3D_python:
        cov3D_precomp, cov3D_precomp_mtx = pc.get_covariance(_scales, scaling_modifier, return_cov3D_mtx=True)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    ##################################################################################
    ########################### RaDe-GS Optimization START ###########################
    ##################################################################################

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ## GEOMETRY optimization, rasterize [rendered_image, rendered_alpha, rendered_normal]
    (rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, 
     rendered_median_depth, rendered_alpha, rendered_normal, normals) = rade_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    results.update({
        "rendered_image": rendered_image,
        "opacity_render": rendered_alpha,
        "rendered_normal": rendered_normal,
        "expected_coord": rendered_expected_coord,
        "median_coord": rendered_median_coord,
        "expected_depth": rendered_expected_depth,
        "median_depth": rendered_median_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "loss_reg": loss_reg,})

    ##################################################################################
    ############################ NEILF Optimization START ############################
    ##################################################################################
    # normal = pc.get_normal_per_vertex(cov3D_precomp_mtx, data.world_view_transform, data.projection_matrix)
    visibility = pc.get_visibility(scene.sample_num, normals, scales=_scales, opacity=opacity)
    
    # base_color = pc.get_base_color
    # roughness = pc.get_roughness
    # metallic = pc.get_metallic
    base_color, roughness, metallic = torch.split(intrinsic_precompute, [3, 1, 1], dim=-1)
    
    features = torch.cat([base_color, normals, roughness, metallic, visibility.mean(-2)], dim=-1)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image2, rendered_opacity2, rendered_depth,
    rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii2) = r3d_rasterizer(
        means3D=means3D.detach(),
        means2D=means2D.detach(),
        shs=shs,
        colors_precomp=colors_precomp.detach(),
        opacities=opacity.detach(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp.detach(),
        features=features,
    )
    rendered_base_color, rendered_normal2, rendered_roughness, rendered_metallic, \
        rendered_visibility = rendered_feature.split([3, 3, 1, 1, 1], dim=0)
    
    # formulate roughness
    rmax, rmin = 1.0, 0.04
    rendered_roughness = rendered_roughness * (rmax - rmin) + rmin
    # rendered_roughness = rendered_roughness.mean(0, keepdim=True)
    # rendered_metallic = rendered_metallic.mean(0, keepdim=True)

    # PBR rendering
    rays = scene.get_canonical_rays(data)
    c2w = torch.inverse(data.world_view_transform.T)  # [4, 4]
    view_dirs = -(
        (F.normalize(rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
        .sum(dim=-1)
        .reshape(data.image_height, data.image_width, 3)
    )  # [H, W, 3]
    points = (
        (-view_dirs.reshape(-1, 3) * rendered_median_depth.reshape(-1, 1) + c2w[:3, 3])
        .clamp(min=-scene.cfg.opt.bound, max=scene.cfg.opt.bound)
        .contiguous()
    )  # [HW, 3]
    
    ##################################################################################
    ################################## PBR Rendering #################################
    ##################################################################################
    rendered_maps = [rendered_base_color,  rendered_metallic, rendered_roughness, rendered_normal2, rendered_alpha, rendered_median_depth, rendered_visibility]
    pbr_result = pbr_render(data, scene, rendered_maps, bg_color)
    
    rendered_pbr = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
    rendered_diffuse = pbr_result["diffuse_rgb"].permute(2, 0, 1)  # [3, H, W]
    rendered_specular = pbr_result["specular_rgb"].permute(2, 0, 1)  # [3, H, W]

    results.update({
        "rendered_pbr": rendered_pbr,
        "rendered_image2": rendered_image2,
        "rendered_normal2": rendered_normal2,
        "rendered_pseudo_normal": rendered_pseudo_normal,
        "albedo_map": rendered_base_color, 
        "roughness_map": rendered_roughness, 
        "metallic_map": rendered_metallic,
        "occlusion_map": rendered_visibility,
        "diffuse_map": rendered_diffuse,
        "specular_map": rendered_specular,
    })
    
    return results