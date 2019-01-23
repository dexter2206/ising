"""Installation script for the ising package."""
import argparse
import glob
import os
import subprocess
from setup_helpers import find_cuda_home, get_cuda_version

# The below map translates user-friendly compiler name
FCOMPILER_MAP = {'pgi': 'pg',
                 'intel': 'intelem',
                 'gfortran': 'gnu95'}
FEXECUTABLE_MAP = {'pgi': 'pgfortran', 'intel': 'ifort', 'gfortran': 'gfortran'}

UNSAFE_FLAGS = ['F90FLAGS', 'F90', 'FFLAGS']

def main():
    """Entry point of this script."""
    for key in UNSAFE_FLAGS:
        if key in os.environ:
            del os.environ[key]

    parser = argparse.ArgumentParser()
    parser.add_argument('--fcompiler', choices=['pgi', 'intel', 'gfortran'],
                        help='Fortran compiler to use')
    parser.add_argument('--usecuda', help='whether to also compile CUDA implementation',
                        action='store_true')
    parser.add_argument('--check', help='if present, only verify installation prerequisites',
                        action='store_true')
    parser.add_argument('--debug', help='use debug configuration when compiling fortran code',
                        action='store_true')

    cmd_args = parser.parse_args()

    # Override compiler if --usecuda is provided
    if cmd_args.usecuda:
        cmd_args.fcompiler='pgi'

    if not run_check(cmd_args):
        return

    if cmd_args.check:
        return

    args = ['python', 'setup.py', 'build_ext', '--fcompiler=' + FCOMPILER_MAP[cmd_args.fcompiler]]
    if cmd_args.debug:
        args += ['--debug']
    if cmd_args.usecuda:
        args += ['--usecuda']
    args += ['build', 'bdist_wheel']

    proc = subprocess.Popen(['rm', '-rf', 'dist', 'build'])
    proc.wait()

    proc = subprocess.Popen(args)
    proc.wait()
    if proc.returncode != 0:
        print('ERROR! build process failed.')
        exit(1)

    wheels = glob.glob('./dist/*.whl')
    latest_wheel = max(wheels, key=os.path.getctime)

    proc = subprocess.Popen(['pip', 'install', latest_wheel, '--upgrade'])
    proc.wait()

def run_check(args):

    fcompiler = args.fcompiler
    fexecutable = FEXECUTABLE_MAP[fcompiler]

    proc = subprocess.Popen([fexecutable, '--version'])
    proc.wait()

    if proc.returncode != 0:
        print('FAILURE: Executable {} for fcompiler={} not found.'.format(fexecutable, fcompiler))
        return False

    print('SUCCESS: executable {} found.'.format(fexecutable))

    if args.usecuda:
        try:
            cuda_home = find_cuda_home()
            print('SUCCESS: CUDA home directory found at {}.'.format(cuda_home))
        except ValueError:
            print('FAILURE: nvcc not present in path.')
            return False

        print('SUCCESS: version of CUDA toolkit detected: {}'.format(get_cuda_version()))

    return True

if __name__ == '__main__':
    main()
