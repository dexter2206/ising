"""An example showing that graph mapping can be used in ising."""
from __future__ import print_function
from builtins import range
import logging
from numpy import random
import ising

NUM_SPINS = 26
NUM_STATES = 10
OMP_THREADS = 4

def main(num_spins, num_states, omp_threads):
    """An entry point of the script."""
    graph = {}
    for i in range(num_spins):
        graph[(i, i)] = random.rand() * 10 - 5
        for j in range(i, num_spins):
            graph[(i, j)] = random.rand() * 10 - 5
    print('Running ising GPU implementation')
    gpu = ising.search(graph,
                       num_states=num_states,
                       method='GPU',
                       show_progress=True,
                       chunk_size=20)
    print('Running ising CPU implementation')
    cpu = ising.search(graph,
                       num_states=num_states,
                       method='CPU',
                       show_progress=True,
                       chunk_size=20,
                       omp_threads=omp_threads)
    sep = '+' + '+'.join(20 * '-' for _ in range(2)) + '+'
    print('\n' + sep)
    print('|{0:20}|{1:20}|'.format(' ising.search (GPU)', ' ising.search (CPU)'))
    print(sep)
    for gpu_en, cpu_en in zip(gpu.energies, cpu.energies):
        print('|{0:20.5f}|{1:20.5f}|'.format(gpu_en, cpu_en))

if __name__ == '__main__':
    main(NUM_SPINS, NUM_STATES, OMP_THREADS)
