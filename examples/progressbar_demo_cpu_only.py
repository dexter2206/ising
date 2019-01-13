"""An example usage of ising package with progressbar - with CPU Implementation only."""
from __future__ import print_function
from numpy import random
import ising
from itertools import islice

NUM_SPINS = 32
NUM_STATES = 2 ** 30
OMP_THREADS = 10

def main(num_spins, num_states, omp_threads):
    """An entry point of the script."""
    graph = random.rand(num_spins, num_spins) * 10 - 5 # pylint: disable=no-member
    print('Running ising.search CPU implementation')
    cpu = ising.search(graph,
                       num_states=num_states,
                       method='CPU',
                       show_progress=False,
                       energies_only=False,
                       omp_threads=omp_threads)
    sep = '+' + 22 * '-' + '+'
    print('\n' + sep)
    print('| {0:20} |'.format('Energy'))
    print(sep)
    for cpu_en in islice(cpu.energies, 10):
        print('|{0:22.5f}|'.format(cpu_en))

if __name__ == '__main__':
    main(NUM_SPINS, NUM_STATES, OMP_THREADS)
