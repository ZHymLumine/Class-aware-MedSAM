# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    extensions = []
    ext_name = '_ext'
    op_files = glob.glob('./layers/csrc/*')
    print(op_files)
    include_path = os.path.abspath('./layers/cinclude')

    extensions.append(CUDAExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path]
    ))

    return extensions


if __name__ == "__main__":
    setup(
        name='medsam',
        version='0.0.1',
        author="Yiming Zhang",
        description='vision transformer with progressive sampling',
        python_requires=">=3.9",
        install_requires=["monai", "matplotlib", "scikit-image", "SimpleITK>=2.2.1", "nibabel", "tqdm", "scipy",
                          "einops"],
        packages=find_packages(exclude="notebooks"),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        extras_require={
            "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
            "dev": ["flake8", "isort", "black", "mypy"],
        },
        zip_safe=False
    )
