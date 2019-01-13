"""Common funcionallities for bruteforce module."""
from builtins import range
from collections import namedtuple, Iterable, Sized, Mapping
from itertools import product
from future import utils
import logging
import numpy

EnergiesAndStates = namedtuple('EnergiesAndStates', ['states', 'energies'])

def decode_state(state_repr, no_spins, labels):
    """Decode integer representation of state into array +/- 1 array.

    :param state_repr: integer to be decoded
    :type state_repr: int
    :param no_bits: how many bits to decode. Note that this parameter is
     neccessary because the leading zeros matter.
    :type no_bits: int
    :returns: array of len `no_bits `consisting of -1 and 1
     respectively if corresponding bit was unset/set. Bits are taken in
     higher -> lower order, so for instance number (01101)b gets converted
     to [1, -1, 1, 1, -1].
    :rtype: :py:class:`numpy.ndarray`
    """
    state = {}

    for i in range(no_spins):
        state[labels[no_spins-(i+1)]] = 1 if state_repr % 2 else -1
        state_repr //= 2
    return state

def ising_to_qubo(ham):
    qubo = numpy.zeros(ham.shape)
    constant = 0
    for i in range(qubo.shape[0]):
        qubo[i, i] = 2 * ham[i, i]
        constant +=  ham[i, i]
        for j in range(qubo.shape[0]):
            if i != j:
                low, high = sorted((i, j))
                constant -= ham[low, high] * 0.5
                qubo[i, i] -= 2 * ham[low, high]
                qubo[low, high] += ham[low, high] * 2
    return qubo, constant
