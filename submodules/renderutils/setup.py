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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="renderutils",
    packages=["renderutils"],
    ext_modules=[
        CUDAExtension(
            name="renderutils._C",
            sources=[
                'c_src/mesh.cu',
                'c_src/loss.cu',
                'c_src/bsdf.cu',
                'c_src/normal.cu',
                'c_src/cubemap.cu',
                'c_src/common.cpp',
                'c_src/torch_bindings.cpp'
            ],
            extra_compile_args={
                "nvcc": [
                    '-DNVDR_TORCH',
                    '-lcuda', 
                    '-lnvrtc'
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
