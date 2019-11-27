import cython
from libcpp cimport bool
from ctypes import c_float, c_double
import progressbar


cdef void callback(int chunk, void* progressbar):
    if progressbar != NULL:
        (<object>progressbar).update(chunk)

cdef extern from "search.cuh":
    ctypedef void (*callback_function)(int, void*);
    cdef void find_lowest[T](T*, int, int, T*, long int*, int, int, int, void*, callback_function)
    cdef void find_lowest_energies_only[T](T*, int, int, T*, int, int, int, void*, callback_function)
    cdef void getGPUMemInfo(unsigned long int* free, unsigned long int* total)

def cy_find_lowest_float(
    float[:,:] Q,
    int num_spins,
    int chunk_exponent,
    float[:] energies_out,
    long int[:] states_out,
    int num_states,
    int block_size,
    int grid_size,
    bool show_progress
):
    if show_progress:
        pbar = progressbar.ProgressBar(max_value=2 ** (num_spins-chunk_exponent))
        pbar.update(0)
    else:
        pbar = None
	
    find_lowest[float](
        &Q[0,0],
        num_spins,
        chunk_exponent,
        &energies_out[0],
        &states_out[0],
        num_states,
	grid_size,
	block_size,
	<void *> pbar,
	callback	
    )

    if pbar:
        pbar.finish()
	
def cy_find_lowest_double(
    double[:,:] Q,
    int num_spins,
    int chunk_exponent,
    double[:] energies_out,
    long int[:] states_out,
    int num_states,
    int block_size,
    int grid_size,
    bool show_progress
):
    if show_progress:
        pbar = progressbar.ProgressBar(max_value=2 ** (num_spins-chunk_exponent))
        pbar.update(0)
    else:
        pbar = None
	
    find_lowest[double](
        &Q[0,0],
        num_spins,
        chunk_exponent,
        &energies_out[0],
        &states_out[0],
        num_states,
	grid_size,
	block_size,
	<void *> pbar,
	callback	
    )

    if pbar:
        pbar.finish()

def cy_find_lowest_energies_only_float(
    float[:,:] Q,
    int num_spins,
    int chunk_exponent,
    float[:] energies_out,
    int num_states,
    int block_size,
    int grid_size,
    bool show_progress
):
    if show_progress:
        pbar = progressbar.ProgressBar(max_value=2 ** (num_spins-chunk_exponent))
        pbar.update(0)
    else:
        pbar = None
	
    find_lowest_energies_only[float](
        &Q[0,0],
        num_spins,
        chunk_exponent,
        &energies_out[0],
        num_states,
	grid_size,
	block_size,
	<void *> pbar,
	callback	
    )

    if pbar:
        pbar.finish()

def cy_find_lowest_energies_only_double(
    double[:,:] Q,
    int num_spins,
    int chunk_exponent,
    double[:] energies_out,
    int num_states,
    int block_size,
    int grid_size,
    bool show_progress
):
    if show_progress:
        pbar = progressbar.ProgressBar(max_value=2 ** (num_spins-chunk_exponent))
        pbar.update(0)
    else:
        pbar = None
	
    find_lowest_energies_only[double](
        &Q[0,0],
        num_spins,
        chunk_exponent,
        &energies_out[0],
        num_states,
	grid_size,
	block_size,
	<void *> pbar,
	callback
    )

    if pbar:
        pbar.finish()
	
def cy_get_gpu_mem_info():
    cdef long unsigned int free
    cdef long unsigned int total
    getGPUMemInfo(&free, &total)
    return free, total