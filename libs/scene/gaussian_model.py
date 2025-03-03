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
import cv2
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import trimesh
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2
from libs.utils.sh_utils import RGB2SH
from libs.utils.graphics_utils import BasicPointCloud, sample_incident_rays, geom_transform_points
from libs.utils.general_utils import strip_symmetric, build_scaling_rotation
from libs.scene.appearance_network import AppearanceNetwork
from libs.utils.bvh import RayTracer
from libs.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_cov3D_act=False):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            if return_cov3D_act:
                return symm, actual_covariance
            else:
                return symm
            
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, cfg, render_type='3dgs'):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.xyz_gradient_accum_abs_max = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
    
        # appearance network and appearance embedding
        # this module is adopted from GOF
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)
        self.filter_3D = torch.empty(0)

        # rendering parameters
        self.render_type = render_type
        self.use_pbr = render_type in ['neilf']
        if self.use_pbr:
            # self._base_color = torch.empty(0)
            # self._metallic = torch.empty(0)
            # self._roughness = torch.empty(0)            
            self.intrinsic_dim = cfg.feature_dim
            self._intrinsic = torch.empty(0)
            
        self.setup_functions()

    def clone(self):
        cloned = GaussianModel(self.cfg, self.render_type)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      "_appearance_embeddings",
                      "appearance_network",
                      "filter_3D"
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_normal",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity"]

        if self.use_pbr:
            # parameters.extend([
            #               "_base_color",
            #               "_metallic",
            #               "_roughness"])
            parameters.extend([
                "_intrinsic",
            ])
        
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        captured_list = [
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.xyz_gradient_accum_abs_max,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.filter_3D,
            self.appearance_network.state_dict(),
            self._appearance_embeddings
        ]
        if self.use_pbr:
            # captured_list.extend([
            #     self._base_color,
            #     self._metallic,
            #     self._roughness,
            # ])
            captured_list.extend([
                self._intrinsic,
            ])
        return captured_list
    
    def restore(self, model_args, training_args,
                is_training=True, restore_optimizer=True):
        (self.active_sh_degree, 
        self._xyz,
        self._normal,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        xyz_gradient_accum_abs_max,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.filter_3D,
        app_dict,
        _appearance_embeddings) = model_args[:18]

        if len(model_args) > 18 and self.use_pbr:
            # (self._base_color, 
            # self._roughness, 
            # self._metallic,) = model_args[17:]
            self._intrinsic = model_args[18]

        if is_training:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
            self.xyz_gradient_accum_abs_max = xyz_gradient_accum_abs_max
            self.denom = denom
            self.appearance_network.load_state_dict(app_dict)
            self._appearance_embeddings = _appearance_embeddings
            if restore_optimizer:
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    raise ValueError("Cannot restore optimizer state dict")

    ########## Deprecated ##########
    @torch.no_grad()
    def prefix_for_geometry(self, cov3D_precomp_act, view_transform, proj_transform):
        full_proj = proj_transform @ view_transform
        cov3D_precomp_act_homo = torch.eye(4).unsqueeze_(0).repeat([cov3D_precomp_act.shape[0], 1, 1]).to(cov3D_precomp_act)
        cov3D_precomp_act_homo[:, :3, :3] = cov3D_precomp_act
        cov_2d = (full_proj @ cov3D_precomp_act_homo @ full_proj.T)[:, :3, :3] # [N_pts, 3, 3]

        v_pre = torch.tensor([0, 0, 1]).reshape(3,1).to(cov_2d) # [3, 1]
        v_pre_T_mut_cov_2d = v_pre.T @ cov_2d.inverse()
        q_hat = v_pre_T_mut_cov_2d / (v_pre_T_mut_cov_2d @ v_pre + 1e-8) # [N_pts, 1, 3]

        return q_hat
    
    ########## Deprecated ##########
    @torch.no_grad()
    def get_normal_per_vertex(self, cov3D_precomp_mtx, view_transform, proj_transform):
        # precompute q_hat for normal computation
        q_hat = self.prefix_for_geometry(cov3D_precomp_mtx, view_transform, proj_transform)
        
        # [TODO: 有bug, 需修改] -> fixed.
        # construct normal vectors, equation (10) in the paper # TODO: check the sign
        q_one =  torch.cat([q_hat[..., :2], torch.ones_like(q_hat[..., 2:])], dim=-1) # [N, 1, 3]
        n_pre = -torch.permute(q_one, [0, 2, 1]) # [N, 1, 3]
        normal_cam = proj_transform[:3, :3].T @ n_pre # [N, 3, 1]
        return torch.nn.functional.normalize(normal_cam.squeeze(-1))
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_normal(self):
        return torch.nn.functional.normalize(self._normal)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        scales = torch.sqrt(scales_after_square)
        return scales, opacity * coef[..., None]
    
    # @property
    # def get_base_color(self):
    #     return self.base_color_activation(self._base_color)
    
    # @property
    # def get_roughness(self):
    #     return self.roughness_activation(self._roughness)
    
    # @property
    # def get_metallic(self):
    #     return self.metallic_activation(self._metallic)

    # @property
    # def get_brdf(self):
    #     return torch.cat([self.get_base_color, self.get_roughness, self.get_metallic], dim=-1)
    
    @property
    def get_intrinsic(self):
        return self._intrinsic
    
    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'normal', 'features_dc', 'features_rest','scaling', 'rotation', 'opacity']
        if self.use_pbr:
            # attribute_names.extend(['base_color', 'roughness', 'metallic',])
            attribute_names.extend(['intrinsic'])
            
        return attribute_names
    
    def get_by_names(self, names):
        if len(names) == 0:
            return None
        fs = []
        for name in names:
            fs.append(getattr(self, "get_" + name))
        return torch.cat(fs, dim=-1)

    def split_by_names(self, features, names):
        results = {}
        last_idx = 0
        for name in names:
            current_shape = getattr(self, "_" + name).shape[1]
            results[name] = features[last_idx:last_idx + current_shape]
            last_idx += getattr(self, "_" + name).shape[1]
        return results
    
    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    def get_covariance(self, scales = None, scaling_modifier = 1, return_cov3D_mtx=False):
        if hasattr(self, 'rotation_precomp'):
            if scales is not None:
                return self.covariance_activation(scales, scaling_modifier, self.rotation_precomp, return_cov3D_mtx)
            else:
                return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp, return_cov3D_mtx)
        else:
            if scales is not None:
                return self.covariance_activation(scales, scaling_modifier, self._rotation, return_cov3D_mtx)
            else:
                return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, return_cov3D_mtx)

    def get_inverse_covariance(self, scales = None, scaling_modifier=1, return_cov3D_mtx=False):
        if hasattr(self, 'rotation_precomp'):
            if scales is not None:
                return self.covariance_activation(1/scales, 1/scaling_modifier, self.rotation_precomp, return_cov3D_mtx)
            else:
                return self.covariance_activation(1/self.get_scaling, 1/scaling_modifier, self.rotation_precomp, return_cov3D_mtx)
        else:
            if scales is not None:
                return self.covariance_activation(1/scales, 1/scaling_modifier, self._rotation, return_cov3D_mtx)
            else:
                return self.covariance_activation(1/self.get_scaling, 1/scaling_modifier, self._rotation, return_cov3D_mtx)

    @torch.no_grad()
    def reset_3D_filter(self):
        xyz = self.get_xyz
        self.filter_3D = torch.zeros([xyz.shape[0], 1], device=xyz.device)
    
    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:
            # focal_x = float(camera.intrinsic[0,0])
            # focal_y = float(camera.intrinsic[1,1])
            W, H = camera.image_width, camera.image_height
            focal_x = W / (2 * math.tan(camera.FoVx / 2.))
            focal_y = H / (2 * math.tan(camera.FoVy / 2.))

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x
        
        if valid_points.sum() > 0:
            distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]


    @torch.no_grad()
    def compute_partial_3D_filter(self, cameras):
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        point_num = xyz.shape[0]
        current_filter = self.filter_3D.shape[0]
        addition_xyz_num = point_num - current_filter
        if addition_xyz_num == 0:
            return
        addition_xyz = xyz[current_filter:]
        distance = torch.ones((addition_xyz_num), device=xyz.device) * 100000.0
        valid_points = torch.zeros((addition_xyz_num), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:
            # focal_x = float(camera.intrinsic[0,0])
            # focal_y = float(camera.intrinsic[1,1])
            W, H = camera.image_width, camera.image_height
            focal_x = W / (2 * math.tan(camera.FoVx / 2.))
            focal_y = H / (2 * math.tan(camera.FoVy / 2.))

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = addition_xyz @ R + T[None, :]
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = torch.cat([self.filter_3D,filter_3D[..., None]])

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        normal = torch.ones_like(fused_point_cloud)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            # base_color = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
            # roughness = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * 0.5
            # metallic = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * 0.5

            # self._base_color = nn.Parameter(base_color.requires_grad_(True))
            # self._roughness = nn.Parameter(roughness.requires_grad_(True))
            # self._metallic = nn.Parameter(metallic.requires_grad_(True))
            intrinsic = torch.ones((fused_point_cloud.shape[0], 1, self.intrinsic_dim), dtype=torch.float, device="cuda")
            self._intrinsic = nn.Parameter(intrinsic.transpose(1, 2).contiguous().requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
        ]

        if self.use_pbr:
            # l.extend([
            #     {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
            #     {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            #     {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"}
            # ])
            l.extend([
                {'params': [self._intrinsic], 'lr': training_args.intrinsic_lr, "name": "intrinsic"},
            ])
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        
        if self.use_pbr:
            # for i in range(self._base_color.shape[1]):
            #     l.append('base_color_{}'.format(i))
            # for i in range(self._roughness.shape[1]):
            #     l.append('roughness_{}'.format(i))
            # for i in range(self._metallic.shape[1]):
            #     l.append('metallic_{}'.format(i))
            for i in range(self._intrinsic.shape[1]*self._intrinsic.shape[2]):
                l.append('intrinsic_{}'.format(i))
            
        return l

    def save_ply(self, path):
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()

        attributes_list = [xyz, normal, f_dc, f_rest, opacities, scale, rotation, filter_3D]
        if self.use_pbr:
            # attributes_list.extend([
            #     self._base_color.detach().cpu().numpy(),
            #     self._roughness.detach().cpu().numpy(),
            #     self._metallic.detach().cpu().numpy()
            # ])
            attributes_list.extend([
                self._intrinsic.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def get_tetra_points(self):
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3. # TODO test

        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # mask = (opacity > 0.1).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        return vertices, vertices_scale
    
    @torch.no_grad()
    def get_truc_tetra_points(self, cameras, depth_truc):
        xyz = self.get_xyz
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        for camera in cameras:
            # focal_x = float(camera.intrinsic[0,0])
            # focal_y = float(camera.intrinsic[1,1])
            W, H = camera.image_width, camera.image_height
            focal_x = W / (2 * math.tan(camera.FoVx / 2.))
            focal_y = H / (2 * math.tan(camera.FoVy / 2.))

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            
            # project to screen space
            valid_depth = (xyz_cam[:, 2] > 0.2) * (xyz_cam[:, 2] < depth_truc)  # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)

            valid_points = torch.logical_or(valid_points, valid)

        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3. # TODO test
        xyz = xyz[valid_depth]
        scale = scale[valid_depth]
        rots = rots[valid_depth]
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        return vertices, vertices_scale
    
    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad()
    def get_visibility(self, sample_num, gaussians_normal, scales=None, opacity=None):
        raytracer = RayTracer(self.get_xyz, self.get_scaling if scales is None else scales, self.get_rotation, self.rotation_precomp)
        gaussians_xyz = self.get_xyz
        gaussians_inverse_covariance = self.get_inverse_covariance(scales)
        gaussians_opacity = self.get_opacity[:, 0] if opacity is None else opacity[:, 0]
        
        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in range(0, gaussians_xyz.shape[0], chunk_size):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False, sample_num)
            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility)
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        # incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        # incident_areas_result = torch.cat(incident_areas_results, dim=0)
        # self._visibility_tracing = incident_visibility_result
        # self._incident_dirs = incident_dirs_result
        # self._incident_areas = incident_areas_result

        return incident_visibility_result

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 1, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==self.feature_dim - 1
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 1, self.feature_dim - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        
        # self.active_sh_degree = self.max_sh_degree

        if self.use_pbr:
            # # TODO:  complete the loading of PBR parameters
            # base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
            # base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
            # base_color = np.zeros((xyz.shape[0], len(base_color_names)))
            # for idx, attr_name in enumerate(base_color_names):
            #     base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
            # metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

            # self._base_color = nn.Parameter(
            #     torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
            # self._roughness = nn.Parameter(
            #     torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
            # self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
            
            # TODO: check if the loading of PBR parameters is correct
            intrinsic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("intrinsic")]
            intrinsic_names = sorted(intrinsic_names, key=lambda x: int(x.split('_')[-1]))
            intrinsic = np.zeros((xyz.shape[0], len(intrinsic_names)))
            for idx, attr_name in enumerate(intrinsic_names):
                intrinsic[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._intrinsic = nn.Parameter(torch.tensor(intrinsic, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_pbr:
            # self._base_color = optimizable_tensors["base_color"]
            # self._roughness = optimizable_tensors["roughness"]
            # self._metallic = optimizable_tensors["metallic"]
            self._intrinsic = optimizable_tensors["intrinsic"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, 
                              new_rotation, new_intrinsic=None):
        d = {"xyz": new_xyz,
             "normal": new_normal,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation}
        
        if self.use_pbr:
            d.update({
                "intrinsic": new_intrinsic
            })
        extension_num = new_xyz.shape[0]
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            # self._base_color = optimizable_tensors["base_color"]
            # self._roughness = optimizable_tensors["roughness"]
            # self._metallic = optimizable_tensors["metallic"]
            self._intrinsic = optimizable_tensors["intrinsic"]

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self.get_normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        args = [new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        if self.use_pbr:
            # new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
            # new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            # new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
            # args.extend([
            #     new_base_color,
            #     new_roughness,
            #     new_metallic
            # ])
            new_intrinsic = self._intrinsic[selected_pts_mask].repeat(N, 1, 1)
            args.extend([
                new_intrinsic
            ])

        self.densification_postfix(*args)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_normal = self.get_normal[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        args = [new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities,
                new_scaling, new_rotation]
        if self.use_pbr:
            # new_base_color = self._base_color[selected_pts_mask]
            # new_roughness = self._roughness[selected_pts_mask]
            # new_metallic = self._metallic[selected_pts_mask]

            # args.extend([
            #     new_base_color,
            #     new_roughness,
            #     new_metallic
            # ])
            new_intrinsic = self._intrinsic[selected_pts_mask]
            args.extend([
                new_intrinsic
            ])
        
        self.densification_postfix(*args)

    def densify_and_prune(self, opt, scene, max_screen_size):
        extent = scene.cameras_extent

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        # self.densify_and_clone(grads, max_grad, grads_abs, Q, grads_normal, max_grad_normal, extent)
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        
        torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
        