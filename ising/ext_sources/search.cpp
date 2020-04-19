#include <math.h>
#include <stdio.h>
#include <select.h>
#include <Python.h>

typedef int (*callback_function)(int, void*);

template <typename T>
T energy(int64_t state_repr, T* Q, int num_bits)
{
    int i, j;
    short int bit;
    T energy = 0;

    for(i = 0; i < num_bits; i++) {
        bit = (state_repr >> i) & 1;
        if(bit) {
            for(j = i; j < num_bits; j++) {
                energy -= Q[i * num_bits + j] * bit * ((state_repr >> j) & 1);
            }
        }
    }
    return energy;
}

template <typename T>
void find_lowest(
    T* Jh,
    int num_bits,
    int chunk_exponent,
    T* energies_out,
    long int* states_out,
    int num_states,
    void* user_data,
    callback_function callback)
{
    int chunk_size = pow(2, chunk_exponent);

    T* energies = new T[chunk_size];
    long int* states = new long int[chunk_size];
    long int state_repr;

    if(num_states > chunk_size) {
        num_states = chunk_size;
    }

    T* lowest_energies = new T[num_states * 2];
    long int* lowest_states = new long int[num_states * 2];

    for(long int m=0; m < pow(2, num_bits - chunk_exponent); m++) {
        if(callback != NULL) {
            if(callback(m, user_data) == -1) return;
        }

        #pragma omp parallel for private(state_repr)
        for(long int k=0; k < chunk_size; k++) {
            state_repr = k + m * chunk_size;
            energies[k] = energy(state_repr, Jh, num_bits);
            states[k] = state_repr;
        }

        top_k_int_by_key(states, energies, chunk_size, chunk_size-num_states);
        if(m == 0) {
            memcpy(lowest_energies, energies, num_states * sizeof(T));
            memcpy(lowest_states, states, num_states * sizeof(long int));
        } else {
            memcpy(lowest_energies + num_states, energies, num_states * sizeof(T));
            memcpy(lowest_states + num_states, states, num_states * sizeof(long int));
            top_k_int_by_key(lowest_states, lowest_energies, 2 * num_states, num_states);
        }
    }
    memcpy(energies_out, lowest_energies, num_states * sizeof(T));
    memcpy(states_out, lowest_states, num_states * sizeof(long int));
    delete energies;
    delete states;
    delete lowest_energies;
    delete lowest_states;
}

template <typename T>
void find_lowest_energies_only(
    T* Jh,
    int num_bits,
    int chunk_exponent,
    T* energies_out,
    int num_states,
    void* user_data,
    callback_function callback)
{
    int chunk_size = pow(2, chunk_exponent);

    T* energies = new T[chunk_size];
    long int state_repr;

    if(num_states > chunk_size) {
        num_states = chunk_size;
    }

    T* lowest_energies = new T[num_states * 2];

    for(int m=0; m < pow(2, num_bits - chunk_exponent); m++) {
        if(callback != NULL) {
            if(callback(m, user_data) == -1) return;
        }
        #pragma omp parallel for private(state_repr)
        for(int k=0; k < chunk_size; k++) {
            state_repr = k + m * chunk_size;
            energies[k] = energy(state_repr, Jh, num_bits);
        }

        top_k(energies, chunk_size, chunk_size-num_states);

        if(m == 0) {
            memcpy(lowest_energies, energies, num_states * sizeof(T));
        } else {
            memcpy(lowest_energies + num_states, energies, num_states * sizeof(T));
            top_k(lowest_energies, 2 * num_states, num_states);
        }
    }
    memcpy(energies_out, lowest_energies, num_states * sizeof(T));
    delete energies;
    delete lowest_energies;
}

