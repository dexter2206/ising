"""Ising: a package for exactly solving abritrary Ising model instances using exhaustive search."""
from pathlib import Path
import os
from typing import Iterable, Optional
from setuptools import setup
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def read_file(path) -> str:
    """Read whole file."""
    with open(path) as f:
        return f.read()


def find_executable(name, executables_paths: Iterable[Path]) -> Optional[Path]:
    """Find a file in a search path

    Adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    for path in executables_paths:
        bin_path = path.joinpath(name)
        if bin_path.exists():
            return bin_path.absolute()
    return None


def locate_cuda() -> dict:
    """Locate the CUDA environment on the system end return its configuration.

    The returned dictionary is either empty, indicating that CUDA toolkit was not
    found, or contains keys "name", "nvcc", "include", "lib64" that map to the
    absolute paths of those directories.

    This starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding "nvcc" executable in the PATH.

    This is a slightly modernized version of the function here:
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/setup.py
    """
    if "CUDAHOME" in os.environ:
        cuda_home = Path(os.environ["CUDAHOME"])
        nvcc_path = cuda_home.joinpath("bin", "nvcc")
    else:
        nvcc_path = find_executable("nvcc", (Path(p) for p in os.environ["PATH"].split(os.pathsep)))
        # Contrary to original version we don't raise, but return an empty dict
        if nvcc_path is None:
            return {}
        cuda_home = nvcc_path.parent

    cuda_config = {
        "home": cuda_home,
        "nvcc": nvcc_path,
        "include": cuda_home.joinpath("include"),
        "lib64": cuda_home.joinpath("lib64"),
    }

    for key, path in cuda_config.items():
        if not path.exists:
            # If we reached this point, we found CUDA toolkit but didn't find some of the
            # essential directories. To avoid confusion we raise an error.
            raise EnvironmentError(
                f"The CUDA {key} path could not be located in {path}. \n"
                f"Detected CUDA home directory: {cuda_home}."
            )

    return cuda_config


CUDA_CFG = locate_cuda()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            if not CUDA_CFG:
                raise EnvironmentError(
                    "Tried building GPU based version but no nvcc was found. "
                    "Please report this at our issue tracker."
                )
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA_CFG["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            post_args = extra_postargs["nvcc"]
        else:
            post_args = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, post_args, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def construct_extensions():
    include_dirs = [numpy_include, "ising/ext_sources"]
    cpu_extension = Extension(
        "isingcpu",
        sources=["ising/ext_sources/select.cpp", "ising/ext_sources/cpu_wrapper.pyx"],
        libraries=["stdc++", "omp"],
        language="c++",
        extra_compile_args={
            "nvcc": [],
            "gcc": [
                "-c",
                "-O3",
                "-fPIC",
                "-fopenmp",
                "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP",
            ],
        },
        include_dirs=include_dirs,
    )

    extensions = [cpu_extension]

    if CUDA_CFG:
        include_dirs.append(str(CUDA_CFG.get("include")))
        gpu_extension = Extension(
            "isinggpu",
            sources=[
                "ising/ext_sources/kernels.cu",
                "ising/ext_sources/search.cu",
                "ising/ext_sources/gpu_wrapper.pyx",
            ],
            libraries=["stdc++", "cudart"],
            library_dirs=[str(CUDA_CFG.get("lib64"))],
            language="c++",
            extra_compile_args={
                "nvcc": ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
                "gcc": [],
            },
            include_dirs=include_dirs,
        )
        extensions.append(gpu_extension)
    return extensions


class BuildExtCommand(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


setup(
    use_scm_version=True,
    name="ising",
    description=__doc__,
    long_description=read_file("README.rst"),
    cmdclass={"build_ext": BuildExtCommand},
    setup_requires=["setuptools_scm", "cython"],
    install_requires=["numpy>=0.16.0", "psutil", "progressbar2", "future", "cython"],
    ext_modules=cythonize(construct_extensions()),
    packages=["ising"],
)
