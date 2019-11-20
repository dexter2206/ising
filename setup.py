"""Ising: a Python package for exactly solving abritrary Ising model instances using exhaustive search."""
import os
from os.path import join as pjoin
from setuptools import find_packages # pylint: disable=unused-import
import numpy as np
from numpy.distutils.core import setup, Extension
from numpy.distutils.log import set_verbosity
from setup_helpers import BuildExtCommand, find_cuda_home
from Cython.Build import cythonize
set_verbosity(1)

with open('README.rst') as readme:
    LONG_DESCRIPTION = readme.read()

try:
    find_cuda_home()
    CPP_EXT = '.cu'
except ValueError:
    CPP_EXT = '.cpp'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

CUDA = locate_cuda()


CPU_SEARCH_EXT = Extension('isingcpu',
                           extra_compile_args=[
                               '-fPIC',
                               '-fopenmp',
                               '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP',
                               '-lstdc++'],
                           sources=[
                               './ising/ext_sources/bucketSelectCPU' + CPP_EXT,
                               './ising/ext_sources/bucketselectcpu.f90',
                               './ising/ext_sources/cpucsort' + CPP_EXT,
                               './ising/ext_sources/cpu_thrust_sort.f90',
                               './ising/ext_sources/cpusearch.pyf',
                               './ising/ext_sources/cpusearch.f90'])

GPU_SEARCH_EXT = Extension('isinggpu',
                           sources=[
                               './ising/ext_sources/gpusearch.pyf',
                               './ising/ext_sources/gpucsort.cu',
                               './ising/ext_sources/global.f90',
                               './ising/ext_sources/gpu_thrust_sort.f90',
                               './ising/ext_sources/bucketSelect.cu',
                               './ising/ext_sources/bucketselect.f90',
                               './ising/ext_sources/search.f90',
                               './ising/ext_sources/gpusearch.f90'])

EXTENSIONS = [CPU_SEARCH_EXT, GPU_SEARCH_EXT]


CPU_EXTENSION = Extension(
    "isinggpu",
    sources=[
        "ising/ext_sources/select.cpp",
        "ising/ext_sources/cpu_wrapper.pyx"
    ],
    libraries=["stdc++"],
    language="c++",
    extra_compile_args={
        "nvcc": [],
        "gcc": [
            "-c",
            "-O3",
            "-fPIC",
            "-fopenmp",
            "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP",
        ]
    },
    include_dirs=[numpy_include, CUDA["include"], "cythontest/ext_sources"]
)

EXTENSIONS = [CPU_EXTENSION]

class BuildExtCommand(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


setup(
    use_scm_version=True,
    name='ising',
    description=__doc__,
    long_description=LONG_DESCRIPTION,
    cmdclass={'build_ext': BuildExtCommand},
    setup_requires=['setuptools_scm'],
    install_requires=['numpy>=0.16.0', 'psutil', 'progressbar2', 'future'],
    ext_modules=cythonize(EXTENSIONS),
    packages=['ising']
)
