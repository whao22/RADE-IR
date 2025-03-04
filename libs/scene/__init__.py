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
import math
import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from gs_ir import IrradianceVolumes
from libs.models import GaussianConverter
from libs.utils.pbr import CubemapLight, get_brdf_lut
from libs.scene.gaussian_model import GaussianModel
from libs.dataset import load_dataset
from libs.models.network_utils import VanillaCondMLP

def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec

class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians
        self.use_pbr = gaussians.use_pbr

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

        self.normal_refine = VanillaCondMLP(3, 0, 3, cfg.model.texture.mlp).cuda()
        
        self.opt_params = [
            {"params": self.normal_refine.parameters(), "lr": self.cfg.opt.normal_lr}    
        ]
        
        if self.use_pbr:
            self.sample_num = self.cfg.opt.sample_num
            self.brdf_lut = get_brdf_lut().cuda()
            self.envmap_dirs = get_envmap_dirs()
            bound = self.cfg.opt.bound
            aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
            self.irradiance_volumes = IrradianceVolumes(aabb).cuda()
            self.cubemap = CubemapLight(base_res=256).cuda()
            
            self.opt_params.extend([
                {"params": self.irradiance_volumes.parameters(), "lr": self.cfg.opt.opacity_lr},
                {"params": self.cubemap.parameters(), "lr": self.cfg.opt.opacity_lr},
            ])
    
    def train(self):
        self.converter.train()
        if self.use_pbr:
            self.cubemap.train()
            self.irradiance_volumes.train()
            
    def eval(self):
        self.converter.eval()
        if self.use_pbr:
            self.cubemap.eval()
            self.irradiance_volumes.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

        if self.use_pbr:
            self.light_optimizer = torch.optim.Adam(self.opt_params, lr=self.cfg.opt.light_lr)
            self.light_optimizer.step()
            self.light_optimizer.zero_grad()
        
    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        save_dict = [iteration,
                     self.gaussians.capture(),
                     self.converter.state_dict(),
                     self.converter.optimizer.state_dict(),
                     self.converter.scheduler.state_dict(),]
        
        if self.use_pbr:
            save_dict.extend([self.irradiance_volumes.state_dict(),
                              self.cubemap.state_dict(),
                              self.light_optimizer.state_dict()])
            
        torch.save(save_dict, self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path, restore_optimizer=True):
        loaded_dict= torch.load(path)
        (first_iter, gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd) = loaded_dict[:5]
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        if restore_optimizer:
            self.converter.optimizer.load_state_dict(converter_opt_sd)
            self.converter.scheduler.load_state_dict(converter_scd_sd)
        
        if self.use_pbr:
            (irradiance_volumes_sd, cubemap_sd, light_optimizer_sd) = loaded_dict[5:]
            self.irradiance_volumes.load_state_dict(irradiance_volumes_sd)
            self.cubemap.load_state_dict(cubemap_sd)
            if restore_optimizer:
                self.light_optimizer.load_state_dict(light_optimizer_sd)

        return first_iter

    def get_canonical_rays(self, ref_camera=None, scale: float = 1.0) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        if ref_camera is None:
            ref_camera = self.train_dataset[0]
        
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()