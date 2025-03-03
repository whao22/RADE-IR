import torch
import torch.nn as nn

from libs.utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from libs.utils.general_utils import build_rotation
from libs.models.network_utils import VanillaCondMLP

class ColorPrecompute(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

    def forward(self, gaussians, camera):
        raise NotImplementedError

class SH2RGB(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        
    def forward(self, gaussians, camera):
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.get('view_noise', 0.)
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp
        
class ColorMLP(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        d_in = cfg.feature_dim

        self.use_xyz = cfg.get('use_xyz', False)
        self.use_cov = cfg.get('use_cov', False)
        self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.latent_dim = cfg.get('latent_dim', 0)

        if self.use_xyz:
            d_in += 3
        if self.use_cov:
            d_in += 6 # only upper triangle suffice
        if self.use_normal:
            d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim
        if self.latent_dim > 0:
            d_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3
        self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)
        self.color_activation = nn.Sigmoid()
        
        d_out2 = 5
        self.mlp2 = VanillaCondMLP(d_in, 0, d_out2, cfg.mlp)
        self.intrinsic_activation = lambda x : torch.sigmoid(x) * 0.9 + 0.099
        
        self.mlp3 = VanillaCondMLP(d_in - cfg.feature_dim + 3, 0, 3, cfg.mlp)
        self.normal_activation = torch.nn.functional.normalize
        
    def compose_input(self, gaussians, camera, features):
        n_points = features.shape[0]
        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            features = torch.cat([features, xyz_norm], dim=1)
        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)
        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
            features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.get('view_noise', 0.)
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            dir_embed = self.sh_embed(dir_pp_normalized)
            features = torch.cat([features, dir_embed], dim=1)
        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(features.shape[0], -1)
            features = torch.cat([features, latent_code], dim=1)

        return features


    def forward(self, gaussians, camera):
        # feature
        features = gaussians.get_features.squeeze(-1) # [num_gaussians, feature_dim]
        features_inp = self.compose_input(gaussians, camera, features)
        output = self.mlp(features_inp)
        color = self.color_activation(output)
        
        # intrinsic
        intrinsic = gaussians.get_intrinsic.squeeze(-1) # [num_gaussians, intrinsic_dim]
        intrinsic_inp = torch.cat([intrinsic, features_inp[:, features.shape[-1]:]], dim=1)
        output2 = self.mlp2(intrinsic_inp)
        intrinsic = self.intrinsic_activation(output2)
        
        # normal
        normal = gaussians.get_normal.squeeze(-1) # [num_gaussians, normal_dim]
        normal_inp = torch.cat([normal, features_inp[:, features.shape[-1]:]], dim=1)
        output3 = self.mlp3(normal_inp)
        normal = self.normal_activation(output3)
        
        return color, intrinsic, normal


def get_texture(cfg, metadata):
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,
    }
    return model_dict[name](cfg, metadata)