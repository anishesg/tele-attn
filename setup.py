from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Collect all .cu source files from src/.
src_dir = os.path.join(os.path.dirname(__file__), "src")
cu_sources = [
    os.path.join(src_dir, f)
    for f in os.listdir(src_dir)
    if f.endswith(".cu")
]

ext = CUDAExtension(
    name="tele_attn._C",
    sources=["csrc/bindings.cpp"] + cu_sources,
    include_dirs=[src_dir],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-use_fast_math",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ],
    },
)

setup(
    name="tele_attn",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
