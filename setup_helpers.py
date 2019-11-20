"""Script with helper routines for setup process."""
from distutils.spawn import find_executable
import json
import os
import re
import subprocess
import sys
from Cython.Distutils import build_ext


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
        else:
            cuda_ver = get_cuda_version()
            for ext in self.extensions:
                if 'isinggpu' in ext.name:
                    ext.extra_f90_compile_args.append('-Mcuda=nordc,cuda'+cuda_ver)
                ext.extra_link_args.append('-Mcuda=cuda'+cuda_ver)
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


def find_cuda_home():
    if 'CUDAHOME' in os.environ:
        cuda_home = os.environ['CUDAHOME']
    else:
        nvcc_path = find_executable('nvcc')
        if not nvcc_path:
            raise ValueError("It appears that you don't have nvcc in your PATH.")
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    return cuda_home

def get_cuda_version():
    nvcc_proc = subprocess.Popen(['nvcc', '--version'], stdout=subprocess.PIPE)
    nvcc_proc.wait()
    nvcc_version_string = nvcc_proc.stdout.read().decode()
    return next(iter(re.search(r'release (\d+\.\d+)', nvcc_version_string).groups()))
