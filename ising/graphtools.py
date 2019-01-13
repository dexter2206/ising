"""Module containing functionallities for intepreting graph informations in various formats."""
from builtins import range
from collections import namedtuple, Mapping
import logging
from future.utils import iteritems
import numpy

IsingProblem = namedtuple('IsingProblem', ['matrix', 'spin_labels'])


def read_graph(graph):
    """Read Hamiltonian and spin labels from given graph.

    :param graph: a definition of Ising problem in one of the following formats:
     - as a Mapping of spin indices (i, j) to the corresponding coupler (or
       spin weight if i == j).
     - as a row-format, i.e. an iterable of (i, j, value) with analogous meaning
       as in Mapping format.
     - as a square matrix (numpy.ndarray or nested lists) such that graph[i, j] corresponds
       to coupling between spins i and j (or a spin weight if i == j).
    :returns: a named tuplew with matrix and spin_labels components. The matrix
     is a square upper triangular 2D numpy.ndarray, upper triangular, encoding
     system Hamiltonian (without lower triangle unneeded for computations).
     The spin_labels is a tuple of integers mapping spins of system represented
     by the matrix into spins into original graph.
    :rtype: IsingProblem
    """
    logger = logging.getLogger('ising')
    if isinstance(graph, Mapping):
        logger.debug('Provided graph is in mapping format.')
        return _ising_from_mapping(graph)
    detected_shape = numpy.ma.shape(graph)
    if len(detected_shape) != 2:
        raise ValueError('Unsupported graph format specified. Consult docs.')
    if detected_shape[1] == 3 and has_possibly_spin_labels(graph):
        return _ising_from_rows(graph)

    if detected_shape[0] == detected_shape[1]:
        return _ising_from_matrix(graph)

    raise ValueError('Unsupported graph format specified. Consult docs.')

def has_possibly_spin_labels(graph):
    """Determine if the firstcolumns of a matrix are spin labels.

    :param graph: nested list of lists/tuples or 2D numpy.array. It is assumed
     but not checked that this matrix has at least two columns.
    :returns: True if first two columns contain nonnegative integers or floats
     that can be safely cast to such integers, False otherwise.
    :rtype: bool
    """
    try:
        for row in graph:
            if int(row[0]) != row[0] or int(row[1]) != row[1]:
                return False
            if row[0] < 0 or row[1] < 0:
                return False
    except ValueError:
        return False
    return True

def _ising_from_mapping(graph_map):
    seen_spins = set()
    for i, j in graph_map:
        seen_spins.add(i)
        seen_spins.add(j)
    system_size = len(seen_spins)
    jh_matrix = numpy.zeros(shape=(system_size, system_size))
    spin_labels = tuple(sorted(seen_spins))
    spin_map = {spin: idx for idx, spin in enumerate(spin_labels)}
    for (i, j), value in iteritems(graph_map):
        first, second = sorted((spin_map[i], spin_map[j]))
        jh_matrix[first, second] += value
    return IsingProblem(jh_matrix, spin_labels)

def _ising_from_rows(graph):
    return _ising_from_mapping({(int(i), int(j)): value for i, j, value in graph})

def _ising_from_matrix(graph):
    graph = numpy.array(graph)
    jh_matrix = numpy.zeros(graph.shape)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            first, second = sorted((i, j))
            jh_matrix[first, second] += graph[i, j]
    return IsingProblem(jh_matrix, tuple(range(jh_matrix.shape[0])))
