"""Setup file for bruteforce package."""
from setuptools import find_packages # pylint: disable=unused-import
from numpy.distutils.core import setup, Extension
from numpy.distutils.log import set_verbosity
from setup_helpers import BuildExtCommand

set_verbosity(1)

CPU_SEARCH_EXT = Extension('isingcpu',
                           sources=['./ising/ext_sources/bucketSelectCPU.cu',
                                    './ising/ext_sources/bucketselectcpu.f90',
                                    './ising/ext_sources/cpucsort.cu',
                                    './ising/ext_sources/cpu_thrust_sort.f90',
                                    './ising/ext_sources/cpusearch.pyf',
                                    './ising/ext_sources/cpusearch.f90'])

GPU_SEARCH_EXT = Extension('isinggpu',
                           sources=['./ising/ext_sources/gpusearch.pyf',
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
    version='0.1.16',
    name='ising',
    cmdclass={'build_ext': BuildExtCommand},
    install_requires=['numpy', 'psutil', 'progressbar2', 'future'],
    ext_modules=EXTENSIONS,
    packages=['ising']
)
