"""An example usage of ising package with progressbar."""
from __future__ import print_function
from numpy import random
import ising

NUM_SPINS = 40
NUM_STATES = 100
OMP_THREADS = 4

def main(num_spins, num_states, omp_threads):
    """An entry point of the script."""
    graph = random.rand(num_spins, num_spins) * 10 - 5 # pylint: disable=no-member
    print('Running search with GPU implementation.')
    gpu = ising.search(graph,
                       num_states=num_states,
                       method='GPU',
#                       chunk_size=26,
                       energies_only=False,
                       show_progress=True)
    print('Running search with CPU implementation.')
    cpu = ising.search(graph,
                       num_states=num_states,
                       method='CPU',
                       show_progress=True,
                       chunk_size=27,
                       energies_only=False,
                       omp_threads=omp_threads)
    sep = '+' + '+'.join(20 * '-' for _ in range(2)) + '+'
    print('\n' + sep)
    print('|{0:20}|{1:20}|'.format(' ising.search (GPU)', ' ising.search (CPU)'))
    print(sep)
    for gpu_en, cpu_en in zip(gpu.energies, cpu.energies):
        print('|{0:20.5f}|{1:20.5f}|'.format(gpu_en, cpu_en))

if __name__ == '__main__':
    main(NUM_SPINS, NUM_STATES, OMP_THREADS)
