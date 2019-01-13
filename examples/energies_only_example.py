"""An example usage of ising package with progressbar."""
from __future__ import print_function
from numpy import random
import ising

NUM_SPINS = 26
NUM_STATES = 10
OMP_THREADS = 4

def main(num_spins, num_states, omp_threads):
    """An entry point of the script."""
    graph = random.rand(num_spins, num_spins) * 10 - 5 # pylint: disable=no-member
    print('Running search with CPU implementation - computing only energies.')
    result_no_states = ising.search(graph,
                                    num_states=num_states,
                                    method='CPU',
                                    show_progress=True,
                                    energies_only=True,
                                    omp_threads=omp_threads,
                                    chunk_size=20)
    print('Running search with CPU implementation - computing both states and energies.')
    result_with_states = ising.search(graph,
                                      num_states=num_states,
                                      method='CPU',
                                      show_progress=True,
                                      chunk_size=20,
                                     omp_threads=omp_threads)
    sep = '+' + '+'.join(20 * '-' for _ in range(2)) + '+'
    print('\n' + sep)
    print('|{0:20}|{1:20}|'.format(' ising.search (GPU)', ' ising.search (CPU)'))
    print(sep)
    for energy1, energy2 in zip(result_no_states.energies, result_with_states.energies):
        print('|{0:20.5f}|{1:20.5f}|'.format(energy1, energy2))

if __name__ == '__main__':
    main(NUM_SPINS, NUM_STATES, OMP_THREADS)
