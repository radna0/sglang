from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _read_version() -> str:
    return "0.0.0"


this_dir = Path(__file__).resolve().parent
csrc = this_dir / "csrc"

ext_modules = [
    CUDAExtension(
        name="fa3_fp8_qscale_ext._C",
        sources=[
            str(csrc / "fa3_fp8_qscale.cpp"),
            str(csrc / "fa3_fp8_qscale_cuda.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
            ],
        },
    )
]

setup(
    name="fa3-fp8-qscale-ext",
    version=_read_version(),
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)

