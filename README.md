# RADE-IR: Animatable and Relightable Avatar Inverse Rendering Based on 3D Gaussian Splatting

<div align="center">
    <div style="font-size: 1.3em; font-weight: bold;">
        <a href="https://arxiv.org/abs/24xx.xxxxx">Paper</a> | 
        <a href="https://whao22.github.io/RADE-IR/">Project Page</a> | 
        <a href="https://github.com/whao22/RADE-IR">Github</a>
    </div>
    <div>
        <!-- <a href="https://arxiv.org/abs/24xx.xxxxx">Hao Wang</a> -->
        <!-- <a href="https://arxiv.org/abs/24xx.xxxxx">Ye Wang</a>,
        <a href="https://arxiv.org/abs/24xx.xxxxx">Qingshan Xu</a>,
        <a href="https://arxiv.org/abs/24xx.xxxxx">Rui Ma</a> -->
    </div>
    <div>
        <image src="assets/overview.png" width="100%">
    </div>
</div>

## Abstract
We present RADE-IR, a novel inverse rendering approach that leverages 3D Gaussians on Mesh (RADE) to synthesize novel views of human avatars from monocular video. RADE is a novel representation of human body shape that captures both the geometry and appearance of the body. We first learn a 3D Gaussian distribution on the 3D mesh of the human body using a novel 3D-to-2D projection-based approach. We then use this distribution to synthesize novel views of the human body by sampling from the distribution and rendering the corresponding 2D images. We evaluate our approach on the ZJU-MoCap and PeopleSnapshot datasets, and demonstrate that it can synthesize novel views of human avatars with high fidelity and realistic appearance.


## Requirements
Please run the following commands to set up the environment:
```Shell
conda create -n RADE-IR python=3.8
conda activate RADE-IR

# install pytorch with cuda 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install cudatoolkit-dev=11.7 -c conda-forge

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# install required packages
pip install -r requirements.txt

# install other packages
cd libs/utils
pip install ./bvh
pip install ./gs-ir
```

## Data preparation
### Prerequisites
Download SMPL v1.0.0 models from [here](https://smpl.is.tue.mpg.de/download.php) and put the `.pkl` files under `libs/utils/smpl/models`.
You may need to remove the Chumpy objects following [here](https://github.com/vchoutas/smplx/tree/main/tools).

### ZJU-MoCap

First download the [ZJU-MoCap](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) dataset and save the raw data under `data/zju-mocap`.

Run the following script to preprocess the dataset:
```Shell
python tools/prepare_zjumocap/preprocess_ZJU-MoCap.py --data-dir ${ZJU_ROOT} --out-dir ${OUTPUT_DIR} --seqname ${SCENE}
```
Change `${SCENE}` to one of CoreView_377, CoreView_386, CoreView_387, CoreView_392, CoreView_393, CoreView_394.

The preprocessed data folder will be in the following structure:
```Shell
├── data/data_prepared
    ├── zju-mocap
        ├── CoreView_393
        ├── CoreView_386
        ├── ...
```

### PeopleSnapshot

Download the [PeopleSnapshot](https://graphics.tu-bs.de/people-snapshot) dataset and put them in `data_prepared`. Download the refined training poses from [here](https://drive.google.com/drive/folders/1tbBJYstNfFaIpG-WBT6BnOOErqYUjn6V?usp=drive_link) and put them in the corresponding subjec directory.

``` SHELL
python tools/prepare_snapshot/prepare_dataset.py --cfg tools/prepare_snapshot/male-4-casual.yaml
```

After the preprocessing, the folder will be in the following structure:
```Shell
├── data/data_prepared
    ├── snapshot
        ├── female-3-causual
        ├── female-4-causual
        ├── male-3-causual
        ├── male-4-causual
            ├── animnerf_models
            ├── image
            ├── mask
            ├── ...
```

## Usage

### Training

Run the following command to train from scratch:
```Shell
# ZJU-MoCap
python train.py dataset=zjumocap_393_mono

# PeopleSnapshot
python train.py dataset=ps_male_3 option=iter30k pose_correction=none 
```

### Evaluation

To evaluate the method for a specified subject, run
```Shell
# ZJU-MoCap
python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
# PeopleSnapshot
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3
```

For test on out-of-distribution poses, please download the preprocessed AIST++ and AMASS sequence for subjects in ZJU-MoCap [here](https://drive.google.com/drive/folders/17vGpq6XGa7YYQKU4O1pI4jCMbcEXJjOI?usp=drive_link) 
and extract under the corresponding subject folder `${ZJU_ROOT}/CoreView_${SUBJECT}`.

To animate the subject under out-of-distribution poses, run
```shell
python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

## Relighting

You can relight the synthesized views using the following command by adding the `--hdr=${HDR}` option:
```Shell
# ZJU-MoCap
python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono --hdr=${HDR}
# PeopleSnapshot
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3 --hdr=${HDR}
```
`${HDR}` is one of the `city`, `bridge`, etc.

## Citation

If you find this project useful for your research, please consider citing:
```bibtex
@inproceedings{wang2024RADEir,
    title={{RADE-IR: Inverse Rendering of Human Avatars from Monocular Video via Gaussians-on-Mesh}},
    author={Hao Wang},
    booktitle={arXiv},
    year={2024}
}
```

## Acknowledgements
This project builds upon [RADE](https://github.com/mikeqzy/3dgs-avatar-release), [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian) and [GS-IR](https://github.com/lzhnb/GS-IR). We appreciate the authors for their great work!
