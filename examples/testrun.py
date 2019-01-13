"""An example usage of ising package.

Synopsis:
  This method computes first lowest energies of randomly selected Ising Model using following
  implementations:
    - ising.search with method='CPU'
    - ising.search with method='GPU'
    - pure Python solution (for ilustrating correctness of results)
  You can tweak the below global parameters to influence various aspects of performed
  computations.

  NO_BITS: number of bits in the system. Keep it low, as pure Python solution stores
           temporarily all states in the search space.
  NUM_STATES: How many lowest energies to compute
  LOG_LEVEL: log level to use. If you don't know what it is please leave it at 'INFO'
"""
from __future__ import print_function
from itertools import product
import logging
from numpy import random
import ising

NO_BITS = 10
NUM_STATES = 30
LOG_LEVEL = 'INFO'
LOGFORMAT = '| %(levelname)s | %(asctime)s | %(name)11s | %(message)s'

def search_space(no_bits):
    """Create search space for Python implementation."""
    return list(product([1, -1], repeat=no_bits))

def energy(Jh, state): # pylint: disable=invalid-name
    """Compute energy of the given state."""
    _energy = 0
    for i, first in enumerate(state):
        for j, second in enumerate(state):
            if i != j:
                _energy -= first * Jh[i, j] * second
            elif i == j:
                _energy -= first * Jh[j, j]
    return _energy

def search_py(Jh, num_states): # pylint: disable=invalid-name
    """Find states of lowest energy in the given system."""
    space = search_space(Jh.shape[0])
    energies = [energy(Jh, state) for state in space]
    result = sorted(zip(energies, space))
    return zip(*result[:num_states])

def main(no_bits, num_states, log_level):
    """Entry point of this script."""

    # pylint: disable=invalid-name
    logging.basicConfig(level=log_level, format=LOGFORMAT)
    logger = logging.getLogger('testrun')
    Jh = random.rand(no_bits, no_bits) * 10 - 5 # pylint: disable=no-member
    Jh = Jh + Jh.T
    logger.info('Running pure Python implementation')
    py_result = search_py(Jh, num_states)
    logger.info('Running ising GPU implementation')

    gpu = ising.search(Jh, num_states=num_states, method='GPU', energies_only=True, chunk_size=no_bits)
    logger.info('Running ising CPU implementation')

    cpu = ising.search(Jh, num_states=num_states, method='CPU', energies_only=True, chunk_size=no_bits-2)
    sep = '+' + '+'.join(20 * '-' for _ in range(3)) + '+'
    print(sep)
    print('|{0:20}|{1:20}|{2:20}|'.format(' Python', ' ising (GPU)', ' ising (CPU)'))
    print(sep)
    for py_en, gpu_en, cpu_en in zip(next(iter(py_result)), gpu.energies, cpu.energies):
        print('|{0:20.5f}|{1:20.5f}|{2:20.5f}|'.format(py_en, gpu_en, cpu_en))
    print(num_states)
if __name__ == '__main__':
    main(NO_BITS, NUM_STATES, LOG_LEVEL)
