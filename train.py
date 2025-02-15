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
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from random import randint
import hydra
from omegaconf import OmegaConf
import wandb
import lpips

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from libs.gaussian_renderer import render
from libs.scene import Scene, GaussianModel
from libs.utils.general_utils import fix_random, Evaluator, PSEvaluator, visualize_depth
from libs.models.loss import compute_loss
from libs.utils.loss_utils import l1_loss

def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian, render_type=config.render_type)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        first_iter = scene.load_checkpoint(checkpoint)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=scene.train_dataset)
    
    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map
    kernel_size = dataset.kernel_size
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data = scene.train_dataset[data_idx]
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(data, iteration, scene, pipe, background, kernel_size, 
                            compute_loss=True,
                            require_coord = require_coord and reg_kick_on, 
                            require_depth = require_depth and reg_kick_on)
        loss, loss_dict = compute_loss(iteration, config, dataset, data, render_pkg, scene, loss_fn_vgg, reg_kick_on, require_depth)
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {'iter_time': elapsed}
            log_loss.update({
                'loss/' + k: v for k, v in loss_dict.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background, kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # model.gaussian.delay = -1
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)
                    print("points in the model: ", gaussians.get_xyz.shape[0])
                    
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=scene.train_dataset)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=scene.train_dataset)
                    
            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)

def validation(iteration, testing_iterations, testing_interval, scene : Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
                          {'name': 'train', 'cameras' : [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            examples = []
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, require_coord=True, require_depth=True, compute_loss=False, is_training=False)

                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                rendered_pbr = torch.clamp(render_pkg["rendered_pbr"], 0.0, 1.0)
                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)
                rendered_normal =  (1 - (render_pkg["rendered_normal"] * 0.5 + 0.5)) * opacity_image
                rendered_normal2 =  (1 - (render_pkg["rendered_normal2"] * 0.5 + 0.5)) * opacity_image
                
                rendered_depth = visualize_depth(render_pkg["median_depth"]) * opacity_image
                albedo = render_pkg["albedo_map"]
                roughness = render_pkg["roughness_map"]
                metallic = render_pkg["metallic_map"]
                visibility = render_pkg["occlusion_map"]
                diffuse = render_pkg["diffuse_map"]
                specular = render_pkg["specular_map"]
                
                wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(opacity_image[None],caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/rendering".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(rendered_pbr[None], caption=config['name'] + "_view_{}/rendered_pbr".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(rendered_normal[None], caption=config['name'] + "_view_{}/rendered_normal".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(rendered_normal2[None], caption=config['name'] + "_view_{}/rendered_normal2".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(rendered_depth[None], caption=config['name'] + "_view_{}/rendered_depth".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(albedo[None], caption=config['name'] + "_view_{}/albedo".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(roughness[None], caption=config['name'] + "_view_{}/roughness".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(metallic[None], caption=config['name'] + "_view_{}/metallic".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(visibility[None], caption=config['name'] + "_view_{}/visibility".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(diffuse[None], caption=config['name'] + "_view_{}/diffuse".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(specular[None], caption=config['name'] + "_view_{}/specular".format(data.image_name))
                examples.append(wandb_img)

                l1_test += l1_loss(image, gt_image).mean().double()
                metrics_test = evaluator(image, gt_image)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                wandb.log({config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)

    # set wandb logger
    wandb_name = config.name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    # print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    main()
