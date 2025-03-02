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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import torch
import numpy as np

import open3d as o3d
import open3d.core as o3c
import hydra
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm, trange
from os import makedirs

from libs.gaussian_renderer import render
from libs.utils.general_utils import fix_random
from libs.scene import GaussianModel
from libs.scene import Scene





def mesh_extract(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        geometry_path = os.path.join(config.exp_dir, config.suffix, 'mesh')
        makedirs(geometry_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        times = []
        depth_list = []
        color_list = []
        alpha_thres = 0.5
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background, config.dataset.kernel_size,
                                compute_loss=False, require_coord=True, require_depth=True)
            
            # rendered image
            rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1,2,0)
            color_list.append(np.ascontiguousarray(rendered_img))
            
            # rendered depth
            depth = render_pkg["median_depth"].clone()
            if view.original_mask is not None:
                depth[(view.original_mask < 0.5)] = 0
            depth[render_pkg["opacity_render"]<alpha_thres] = 0
            depth_list.append(depth[0].cpu().numpy())
            
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)
            times.append(elapsed)

        torch.cuda.empty_cache()
        voxel_size = 0.002
        o3d_device = o3d.core.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
                                                attr_dtypes=(o3c.float32,
                                                            o3c.float32,
                                                            o3c.float32),
                                                attr_channels=((1), (1), (3)),
                                                voxel_size=voxel_size,
                                                block_resolution=16,
                                                block_count=50000,
                                                device=o3d_device)

        for color, depth, viewpoint_cam in zip(color_list, depth_list, scene.test_dataset):
            depth = o3d.t.geometry.Image(depth)
            depth = depth.to(o3d_device)
            color = o3d.t.geometry.Image(color)
            color = color.to(o3d_device)
            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
            fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
            intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
            intrinsic = o3d.core.Tensor(intrinsic)
            extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                                                                            depth, 
                                                                            intrinsic,
                                                                            extrinsic, 
                                                                            1.0, 8.0
                                                                        )
            vbg.integrate(
                            frustum_block_coords, 
                            depth, 
                            color,
                            intrinsic,
                            extrinsic,  
                            1.0, 8.0
                        )

        mesh = vbg.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(geometry_path, "recon.ply"), mesh.to_legacy())
        print("done!")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # for mesh extract
    config.dataset.val_views = ['1'] + config.dataset.val_views
    
    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='radea-ir-former',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    mesh_extract(config)

if __name__ == "__main__":
    main()