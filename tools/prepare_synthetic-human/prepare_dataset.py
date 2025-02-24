import os
import cv2
import json
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def main(data_dir, out_dir, seqname):
    # image and mask
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "mask")
    views = os.listdir(image_dir)
    for view in views:
        view_out_dir = os.path.join(out_dir, view)
        image_view_dir = os.path.join(image_dir, view)
        mask_view_dir = os.path.join(mask_dir, view)
        
        os.makedirs(view_out_dir, exist_ok=True)
        os.system(f"cp {image_view_dir}/* {view_out_dir}/")
        os.system(f"cp {mask_view_dir}/* {view_out_dir}/")
        
    # camera params 
    cam_params = {
        "all_cam_names" : [],
    }
    
    extri_file = os.path.join(data_dir, "extri.yml")
    intri_file = os.path.join(data_dir, "intri.yml")
    extri_data = cv2.FileStorage(extri_file, cv2.FILE_STORAGE_READ)
    intri_data = cv2.FileStorage(intri_file, cv2.FILE_STORAGE_READ)
    
    view_names = extri_data.getNode("names")
    for i in range(view_names.size()):
        view_name = view_names.at(i).string()
        cam_params["all_cam_names"].append(view_name)
        cam_params[view_name] = {
            "K": intri_data.getNode(f"K_{view_name}").mat().tolist(),
            "D": intri_data.getNode(f"dist_{view_name}").mat().tolist(),
            "R": extri_data.getNode(f"Rot_{view_name}").mat().tolist(),
            "T": extri_data.getNode(f"T_{view_name}").mat().tolist(),
        }
    extri_data.release()
    intri_data.release()
    with open(os.path.join(out_dir, "cam_params.json"), "w") as f:
        json.dump(cam_params, f)
    
    # motion model
    import sys
    sys.path.append('.')
    from tools.human_body_prior.body_model.body_model import BodyModel
    body_model = BodyModel(bm_path='tools/body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cuda()

    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    params = np.load(os.path.join(data_dir, "motion.npz"))
    for idx in range(len(params['poses'])):
        Rh = params['Rh'][idx]
        Th = params['Th'][idx]
        shapes = params['shapes'][idx]
        poses = params['poses'][idx]
        
        root_orient = R.from_rotvec(np.array(Rh).reshape([-1])).as_matrix()
        trans = np.array(Th).reshape([3, 1])

        betas = np.array(shapes, dtype=np.float32)
        poses = np.array(poses, dtype=np.float32)
        pose_body = poses[3:66].copy()
        pose_hand = poses[66:].copy()

        poses_torch = torch.from_numpy(poses).cuda()
        pose_body_torch = torch.from_numpy(pose_body).cuda()
        pose_hand_torch = torch.from_numpy(pose_hand).cuda()
        betas_torch = torch.from_numpy(betas).cuda()

        new_root_orient = R.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
        new_trans = trans.reshape([1, 3]).astype(np.float32)

        new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
        new_trans_torch = torch.from_numpy(new_trans).cuda()

        # Get shape vertices
        body = body_model(betas=betas_torch[:10].unsqueeze(0))
        minimal_shape = body.v.detach().cpu().numpy()[0]

        # Get bone transforms
        body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch.unsqueeze(0), pose_hand=pose_hand_torch[:6].unsqueeze(0), betas=betas_torch[:10].unsqueeze(0), trans=new_trans_torch)

        from tools.human_body_prior.easymocap.smplmodel import load_model
        body_model_em = load_model(gender='neutral', model_type='smpl')
        verts = body_model_em(poses=poses_torch[:72].unsqueeze(0), shapes=betas_torch[:10].unsqueeze(0), Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()

        vertices = body.v.detach().cpu().numpy()[0]
        new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
        new_trans_torch = torch.from_numpy(new_trans).cuda()

        body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch.unsqueeze(0), pose_hand=pose_hand_torch[:6].unsqueeze(0), betas=betas_torch[:10].unsqueeze(0), trans=new_trans_torch)

        model_params = {
            "minimal_shape": minimal_shape.tolist(),
            "betas": betas[None, :10].tolist(),
            "Jtr_posed": body.Jtr.detach().cpu().numpy()[0].tolist(),
            "bone_transforms": body.bone_transforms.detach().cpu().numpy()[0].tolist(),
            "trans": new_trans[0].tolist(),
            "root_orient": new_root_orient[0].tolist(),
            "pose_body": pose_body.tolist(),
            "pose_hand": pose_hand[:6].tolist(),
        }

        out_filename = os.path.join(model_dir, '{:06d}.npz'.format(idx))
        np.savez(out_filename, **model_params)

    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing for SyntheticHuman.'
    )
    parser.add_argument('--data-dir', type=str, default='data/SyntheticHuman', help='Directory that contains raw SyntheticHuman data.')
    parser.add_argument('--out-dir', type=str, default='data/data_prepared', help='Directory where preprocessed data is saved.')
    parser.add_argument('--seqname', type=str, default='manuel', help='Sequence to process.')
    args = parser.parse_args()
    
    main(os.path.join(args.data_dir, args.seqname), os.path.join(args.out_dir, args.seqname), args.seqname)