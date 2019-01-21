"""Script with helper routines for setup process."""
from distutils.spawn import find_executable
import json
import os
import sys
from numpy.distutils.command.build_ext import build_ext


CONFIG_MAP = {
    'pg': 'config.pgi.json',
    'intelem': 'config.intel.json',
    'gnu95': 'config.gnu.json'}

class BuildExtCommand(build_ext):
    """Build Extensions Command that handles configured used compilers."""

    user_options = build_ext.user_options + [
        ('usecuda', None, 'Whether to also compile CUDA version')]

    def initialize_options(self):
        """Initialize additional options."""
        build_ext.initialize_options(self)
        self.usecuda=False

    def build_extensions(self):
        """Build extensions.

        Since there is no direct way to achieve this from setup file, this method resolves to
        dirty trick of modifying state of isingcpu extension.
        Basically we check what Fortran compiler is in use and then construct extra compiler
        args and list of additional libraries to link.
        """
        fcompiler = self.fcompiler or 'gnu95'
        customize_compiler_for_nvcc(self.compiler)

        try:
            cuda_home = find_cuda_home()
            self.compiler.has_nvcc = True
            for ext in self.extensions:
                ext.include_dirs += [os.path.join(cuda_home, 'include')]
        except ValueError:
            self.compiler.has_nvcc = False

        if not self.usecuda:
            self._remove_gpu_search_ext(self.extensions)

        with open(CONFIG_MAP[fcompiler], 'rt') as cfg_file:
            cfg = json.load(cfg_file)

        if '--debug' in sys.argv:
            base_args = cfg['fortran_compile_args']['debug']
        else:
            base_args = cfg['fortran_compile_args']['optimal']

        cpu_search_ext = [ext for ext in self.extensions if 'isingcpu' in ext.name][0]
        cpu_search_ext.extra_f90_compile_args = cfg['fortran_compile_args']['common'] + base_args
        cpu_search_ext.libraries = cfg['libraries']
#        cpu_search_ext.library_dirs += [find_cuda_lib_dir()]
        cpu_search_ext.extra_link_args += cfg.get('link_args', [])
        build_ext.build_extensions(self)

    @staticmethod
    def _remove_gpu_search_ext(ext_list):
        for ext in ext_list:
            if 'isinggpu' in ext.name:
                ext_list.remove(ext)
                break

def customize_compiler_for_nvcc(compiler):
    """Customize compiler so it can handle .cu files.

    Except the few changes this is taken from https://github.com/rmcgibbo/npcuda-example.
    """
    # tell the compiler it can processes .cu
    compiler.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = compiler.compiler_so
    old_compile = compiler._compile
    auto_depends = getattr(compiler, '_auto_depends', False)
    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if ext == '.cu':
            if 'cpu' in os.path.split(src)[1].lower():
                postargs = ['-c',
                            '-O3',
                            '-Xcompiler',
                            '-fPIC',
                            '-Xcompiler',
                            '-fopenmp',
                            '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP',
                            '-lstdc++',
                            '-lcudart']
            else:
                postargs = ['-Xcompiler',
                            '-fPIC']

            # use the cuda for .cu filese
            compiler.set_executable('compiler_so', 'nvcc')
            compiler._auto_depends = False
        else:
            compiler.set_executable('compiler_so', default_compiler_so)
            postargs = extra_postargs
        old_compile(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        compiler.compiler_so = default_compiler_so
        setattr(compiler, '_auto_depends', auto_depends)
    # inject our redefined _compile method into the class
    compiler._compile = _compile

def find_cuda_home():
    if 'CUDAHOME' in os.environ:
        cuda_home = os.environ['CUDAHOME']
    else:
        nvcc_path = find_executable('nvcc')
        if not nvcc_path:
            raise ValueError("It appears that you don't have nvcc in your PATH.")
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    return cuda_home
