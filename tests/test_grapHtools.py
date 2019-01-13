"""Test cases for ising.graphtools module."""
from ising.graphtools import has_possibly_spin_labels, read_graph
import numpy
import pytest

def test_accepts_two_integral_cols():
    """The has_possibly_spin_labels should correctly accept nonnegative integral columns."""
    rows = [(1, 2, 2.3), (1.0, 2.0, 4), (4, 3, 10)]
    assert has_possibly_spin_labels(rows)

def test_rejects_nonintegral_cols():
    """The has_possibly_spin_labels should return False if it encounters nonintegral float."""
    rows = [(1.5, 2, 3), (2, 3, -1)]
    assert not has_possibly_spin_labels(rows)

def test_rejects_nonpositive_cols():
    """The has_possibly_spin_labels should return False if it encounters negative integers."""
    rows = [(-2, 3, 0), (1, 2, 5)]
    assert not has_possibly_spin_labels(rows)

def test_reads_graph_row_format():
    """The read_graph should correctly construct IsingProblem from row-format graph."""
    rows = [(1, 2, -1), (2, 3, 10.0), (5, 5, 2)]
    expected_mat = numpy.array([[0, -1, 0, 0],
                                [0, 0, 10.0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 2]], dtype='float64')
    problem = read_graph(rows)
    assert numpy.array_equal(problem.matrix, expected_mat)
    assert problem.spin_labels == (1, 2, 3, 5)

def test_reads_graph_map_format():
    """The read_graph should correctly construct IsingProblem from Mapping-encoded graph."""
    graph = {(7, 1): -1.2, (3, 4): 5.0, (4, 1): 0.5, (7, 7): 10.0}
    expected_mat = numpy.array([[0, 0, 0.5, -1.2],
                                [0, 0, 5.0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 10.0]], dtype='float64')
    problem = read_graph(graph)
    assert numpy.array_equal(problem.matrix, expected_mat)
    assert problem.spin_labels == (1, 3, 4, 7)

def test_reads_graph_from_matrix():
    """The read graph should correctly construct IsingProblem from matrix-encoded graph."""
    graph = [[-2.0, 0, 4.5, 2],
             [0, 5.0, 4, 1],
             [2, 3, 0, 0],
             [0, 0, -1, 2.5]]
    expected_mat = numpy.array([[-2.0, 0, 6.5, 2],
                                [0, 5.0, 7, 1],
                                [0, 0, 0, -1],
                                [0, 0, 0, 2.5]], dtype='float64')
    problem = read_graph(graph)
    assert numpy.array_equal(problem.matrix, expected_mat)
    assert problem.spin_labels == (0, 1, 2, 3)
