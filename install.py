"""Installation script for the ising package."""
import argparse
import glob
import os
import subprocess

# The below map translates user-friendly compiler name
FCOMPILER_MAP = {'pgi': 'pg',
                 'intel': 'intelem',
                 'gfortran': 'gnu95'}

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
    parser.add_argument('--debug', help='use debug configuration when compiling fortran code',
                        action='store_true')

    cmd_args = parser.parse_args()
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

if __name__ == '__main__':
    main()
