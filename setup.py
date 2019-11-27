"""Ising: a package for exactly solving abritrary Ising model instances using exhaustive search."""
import os
from os.path import join as pjoin
from setuptools import setup, find_packages # pylint: disable=unused-import
import numpy as np
from distutils.extension import Extension
#from setup_helpers import BuildExtCommand, find_cuda_home, customize_compiler_for_nvcc
from Cython.Distutils import build_ext
from Cython.Build import cythonize

with open('README.rst') as readme:
    LONG_DESCRIPTION = readme.read()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

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


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


CPU_EXTENSION = Extension(
    "isingcpu",
    sources=[
        "ising/ext_sources/select.cpp",
        "ising/ext_sources/cpu_wrapper.pyx"
    ],
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
        ]
    },
    include_dirs=[numpy_include, CUDA["include"], "ising/ext_sources"]
)

GPU_EXTENSION = Extension(
    "isinggpu",
    sources=[
        "ising/ext_sources/kernels.cu",
        "ising/ext_sources/search.cu",
        "ising/ext_sources/gpu_wrapper.pyx"
    ],
    libraries=["stdc++", "cudart"],
    library_dirs = [CUDA['lib64']],
    language="c++",
    extra_compile_args={
        "nvcc": ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
        "gcc": []
    },
    include_dirs=[numpy_include, CUDA["include"], "ising/ext_sources"]
)

EXTENSIONS = [CPU_EXTENSION, GPU_EXTENSION]


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
