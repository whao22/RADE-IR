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
import pdb
import cv2
import torch
import numpy as np
from typing import List
import hydra
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm, trange
from os import makedirs

import torchvision
import torch.nn.functional as F
import nvdiffrast.torch as dr

from libs.gaussian_renderer import render
from libs.utils.general_utils import fix_random
from libs.scene import GaussianModel
from libs.utils.general_utils import Evaluator, PSEvaluator
from libs.scene import Scene


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian, config.render_type)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        if config.hdr is not None:
            hdri = read_hdr(os.path.join("hdr/high_res_envmaps_2k", f"{config.hdr}.hdr"))
            hdri = torch.from_numpy(hdri).cuda()
            res = 256
            # cubemap = CubemapLight(base_res=res).cuda()
            scene.cubemap.base.data = latlong_to_cubemap(hdri, [res, res])

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))
            if config.hdr is not None:
                torchvision.utils.save_image(render_pkg["rendered_pbr"], os.path.join(render_path, f"render_pbr_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        # save envmap
        if config.hdr is not None:
            envmap = scene.cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
            envmap_path = os.path.join(render_path, f"{config.hdr}_envmap_relight.png")
            torchvision.utils.save_image(envmap, envmap_path)
        
        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)



def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian, config.render_type)
        scene = Scene(config, gaussians, config.exp_dir)
        # scene.eval()

        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)
        
        if config.hdr is not None:
            hdri = read_hdr(os.path.join("hdr/high_res_envmaps_2k", f"{config.hdr}.hdr"))
            hdri = torch.from_numpy(hdri).cuda()
            res = 256
            # cubemap = CubemapLight(base_res=res).cuda()
            scene.cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
        
        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, f'renders_{config.hdr}')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background, config.dataset.kernel_size,
                                compute_loss=False, require_coord=True, require_depth=True)
            # pdb.set_trace()
            bbox_mask = render_pkg['opacity_render']
            bbox_mask = bbox_mask.squeeze().detach().cpu().numpy() # (H, W), ndarray
            # x, y, w, h = cv2.boundingRect((bbox_mask*255).astype(np.uint8))
            # image_pred_ = image_pred[y:y+h, x:x+w] / 255
            # image_targ_ = image_targ[y:y+h, x:x+w] / 255
            
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]
            # rendering = rendering[:, y:y+h, x:x+w]

            gt = view.original_image[:3, :, :]
            # gt = gt[:, y:y+h, x:x+w]

            # pdb.set_trace()
            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                         wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            wandb.log({'test_images': wandb_img})

            rendering = torch.cat([rendering, render_pkg['opacity_render']], dim=0)
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))
            if config.hdr is not None:
                torchvision.utils.save_image(render_pkg["rendered_pbr"], os.path.join(render_path, f"render_pbr_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)
        
        # save envmap
        if config.hdr is not None:
            envmap = scene.cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
            envmap_path = os.path.join(render_path, f"{config.hdr}_envmap_relight.png")
            torchvision.utils.save_image(envmap, envmap_path)
        
        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

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
        project='gaussian-splatting-avatar',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()