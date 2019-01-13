"""An example usage of ising package with progressbar."""
from __future__ import print_function
from numpy import random
import ising

NUM_SPINS = 40
NUM_STATES = 10

def main(num_spins, num_states):
    """An entry point of the script."""
    graph = random.rand(num_spins, num_spins) * 10 - 5 # pylint: disable=no-member
    print('Running search with GPU implementation.')
    gpu = ising.search(graph,
                       num_states=num_states,
                       method='GPU',
                       energies_only=False,
                       show_progress=True)

    sep = '+' + 20 * '-'  + '+'
    print('\n' + sep)
    print('|{0:20}|'.format(' Energy'))
    print(sep)
    for gpu_en in gpu.energies:
        print('|{0:20.5f}|'.format(gpu_en))

if __name__ == '__main__':
    main(NUM_SPINS, NUM_STATES)
