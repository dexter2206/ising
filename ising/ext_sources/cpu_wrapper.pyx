import cython
from libcpp cimport bool
from ctypes import c_float, c_double
import progressbar


cdef void callback(int chunk, void* progressbar):
    if progressbar != NULL:
        (<object>progressbar).update(chunk)

cdef extern from "select.h":
    cdef void top_k_int_by_key[T](long int*, T*, int, int)
    cdef void top_k[T](T*, int, int)

cdef extern from "search.cpp":
    ctypedef void (*callback_function)(int, void*);
    cdef void find_lowest[T](T*, int, int, T*, long int*, int, void*, callback_function)
    cdef void find_lowest_energies_only[T](T*, int, int, T*, int, void*, callback_function)

def cy_top_k_int_by_float_key(long int[:] values, float[:] keys, int length, int k):
    top_k_int_by_key[float](&values[0], &keys[0], length, k)

def cy_top_k_int_by_double_key(long int[:] values, double[:] keys, int length, int k):
    top_k_int_by_key[double](&values[0], &keys[0], length, k)

def cy_top_k_float(float[:] array, int length, int k):
    top_k[float](&array[0], length, k)

def cy_top_k_double(double[:] array, int length, int k):
    top_k[double](&array[0], length, k)

def cy_find_lowest_float(
    float[:,:] Q,
    int num_spins,
    int chunk_exponent,
    float[:] energies_out,
    long int[:] states_out,
    int num_states,
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
        <void*>pbar,
        callback,
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
        <void*>pbar,
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
        <void*>pbar,
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
        <void*>pbar,
        callback
    )

    if pbar:
        pbar.finish()
