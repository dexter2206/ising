"""Ising: a Python package for exactly solving abritrary Ising model instances using exhaustive search."""

from setuptools import find_packages # pylint: disable=unused-import
from numpy.distutils.core import setup, Extension
from numpy.distutils.log import set_verbosity
from setup_helpers import BuildExtCommand, find_cuda_home

set_verbosity(1)

with open('README.rst') as readme:
    LONG_DESCRIPTION = readme.read()

try:
    find_cuda_home()
    CPP_EXT = '.cu'
except ValueError:
    CPP_EXT = '.cpp'

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
                               './ising/ext_sources/gpusearch.f90'],
                           extra_link_args=['-Mcuda'],
                           extra_f90_compile_args=['-v', '-Mcuda,nordc'])

EXTENSIONS = [CPU_SEARCH_EXT, GPU_SEARCH_EXT]

setup(
    use_scm_version=True,
    name='ising',
    description=__doc__,
    long_description=LONG_DESCRIPTION,
    cmdclass={'build_ext': BuildExtCommand},
    setup_requires=['setuptools_scm'],
    install_requires=['numpy>=0.16.0', 'psutil', 'progressbar2', 'future'],
    ext_modules=EXTENSIONS,
    packages=['ising']
)
