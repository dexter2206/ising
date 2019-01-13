"""Tests for ising.core module."""
import pytest
from ising import core

@pytest.fixture(name='psutil')
def psutil_factory(mocker):
    return mocker.patch('ising.core.psutil')

@pytest.fixture(name='max_chunk_size')
def max_chunk_size_factory(mocker):
    return mocker.patch('ising.core.max_chunk_size')

@pytest.fixture(name='gpusearch')
def gpusearch_factory(mocker):
    return mocker.patch('ising.core.gpusearch')

def test_choose_method_not_supplied_wo_gpu():
    """The choose_method should return 'cpu' in absence of 'method' kwarg and no GPU ext."""
    assert core.choose_method(None) == 'cpu'
    assert core.choose_method(None, omp_threads=4) == 'cpu'

def test_choose_method_not_supplied_w_gpu():
    """The choose_method should return 'gpu' in absence of 'method' kwarg and GPU ext."""
    assert core.choose_method(object()) == 'gpu'
    assert core.choose_method(object(), chunk_size=4, omp_threads=3) == 'gpu'

def test_choose_method_raises_on_invalid_method():
    """The choose_method should raise ValueError when invalid computation method is supplied."""
    with pytest.raises(ValueError):
        assert core.choose_method(None, method='invalidmethod')

def test_choose_method_supplied_method():
    """The choose_method should return user-supplied method in presence of gpu extension."""
    assert core.choose_method(object(), method='CPU') == 'cpu'
    assert core.choose_method(object(), method='cpu') == 'cpu'
    assert core.choose_method(object(), method='GPU') == 'gpu'
    assert core.choose_method(object(), method='gpu') == 'gpu'

def test_chooses_method_raises_no_gpu():
    """The choose_method should raise in absence of GPU extension when 'gpu' method is specified."""
    with pytest.raises(ValueError):
        assert core.choose_method(None, method='gpu')

    with pytest.raises(ValueError):
        assert core.choose_method(None, method='GPU')

def test_max_chunk_size():
    """The max_chunk_size should correctly compute maximum chunk size to fit in memory."""
    base_mem_in_bytes = 2 * 16 * 2 ** 30
    assert core.max_chunk_size(base_mem_in_bytes + 500 * 2 ** 20) == 29
    assert core.max_chunk_size(base_mem_in_bytes + 600 * 2 ** 20) == 29
    assert core.max_chunk_size(base_mem_in_bytes) == 29
    assert core.max_chunk_size(base_mem_in_bytes + 1024 * 2 ** 20) == 30

def test_computes_chunk_size_with_psutil(max_chunk_size, psutil):
    """The max_chunk_size_for_method function should use psutil to query host memory."""
    psutil.virtual_memory.return_value.available = 100000
    chunk_size = core.max_chunk_size_for_method('cpu')
    psutil.virtual_memory.assert_called_once_with()
    max_chunk_size.assert_called_once_with(100000)
    assert chunk_size == max_chunk_size.return_value

def test_computes_chunk_size_with_get_device_properties(max_chunk_size, gpusearch):
    """The max_chunk_size_for_method function should use get_device_properties to query gpu memy."""
    gpusearch.get_device_properties.return_value = 30000000
    chunk_size = core.max_chunk_size_for_method('gpu')
    gpusearch.get_device_properties.assert_called_once_with()
    max_chunk_size.assert_called_once_with(30000000)
    assert chunk_size == max_chunk_size.return_value

def test_cannot_compute_chunk_size_unknown_method():
    """The max_chunk_size_for_method function should raise ValueError on unknown method."""
    with pytest.raises(ValueError):
        core.max_chunk_size_for_method('gcpu')
