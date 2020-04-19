#include <kernels.cu>
#include <search.cuh>
#include <stdio.h>
#include <select.cu>


template <typename T>
void find_lowest(
    T* Q,
    int N,
    int chunk_exp,
    T* en_out,
    long int* st_out,
    int num_st,
    int grid_size,
    int block_size,
    void* user_data,
    callback_function callback)
{
    int chunk_size = pow(2, chunk_exp);

    T* d_energies;
    T* d_Jh;
    long int* d_states;
    long int idx;

    cudaMalloc((void **) &d_energies, sizeof(T) * chunk_size);
    cudaMalloc((void **) &d_states, sizeof(long int) * chunk_size);
    cudaMalloc((void **) &d_Jh, sizeof(T) * N * N);
    cudaMemcpy(d_Jh, Q, sizeof(T) * N * N, cudaMemcpyHostToDevice);

    if(num_st > chunk_size) {
        num_st = chunk_size;
    }

    T* d_low_en;
    long int* d_low_st;

    cudaMalloc((void **) &d_low_en, sizeof(T) * num_st * 2);
    cudaMalloc((void **) &d_low_st, sizeof(long int) * num_st *2);

    for(long int m=0; m < pow(2, N - chunk_exp); m++) {
        if(callback != NULL) {
            if(callback(m, user_data) == -1) return;
        }
        idx = m * chunk_size;
        search<<<grid_size, block_size>>>(d_Jh, N, chunk_size, d_energies, d_states, idx);

        top_k_int_by_key(d_states, d_energies, chunk_size, chunk_size-num_st, 40, 1024);

        if(m == 0) {
            cudaMemcpy(d_low_en, d_energies, num_st * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_low_st, d_states, num_st * sizeof(long int), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_low_en + num_st, d_energies, num_st * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_low_st + num_st, d_states, num_st * sizeof(long int), cudaMemcpyDeviceToDevice);
            top_k_int_by_key(d_low_st, d_low_en, 2 * num_st, num_st, 40, 1024);
        }
    }
    cudaMemcpy(en_out, d_low_en, num_st * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(st_out, d_low_st, num_st * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_Jh);
    cudaFree(d_energies);
    cudaFree(d_states);
    cudaFree(d_low_en);
    cudaFree(d_low_st);
}


template <typename T>
void find_lowest_energies_only(
    T* Q,
    int N,
    int chunk_exp,
    T* en_out,
    int num_st,
    int grid_size,
    int block_size,
    void * user_data,
    callback_function callback	       )
{
    int chunk_size = pow(2, chunk_exp);

    T* d_energies;
    T* d_Jh;
    long int idx;

    cudaMalloc((void **) &d_energies, sizeof(T) * chunk_size);
    cudaMalloc((void **) &d_Jh, sizeof(T) * N * N);
    cudaMemcpy(d_Jh, Q, sizeof(T) * N * N, cudaMemcpyHostToDevice);

    if(num_st > chunk_size) {
        num_st = chunk_size;
    }

    T* d_low_en;
    long int* d_low_st;

    cudaMalloc((void **) &d_low_en, sizeof(T) * num_st * 2);

    for(int m=0; m < pow(2, N - chunk_exp); m++) {
        if(callback != NULL) {
            if(callback(m, user_data) == -1) return;
        }
        idx = m * chunk_size;
        search_energies_only<<<grid_size, block_size>>>(d_Jh, N, chunk_size, d_energies, idx);

        top_k(d_energies, chunk_size, chunk_size-num_st, 40, 1024);

        if(m == 0) {
            cudaMemcpy(d_low_en, d_energies, num_st * sizeof(T), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_low_en + num_st, d_energies, num_st * sizeof(T), cudaMemcpyDeviceToDevice);
            top_k(d_low_en, 2 * num_st, num_st, 40, 1024);
        }
    }
    cudaMemcpy(en_out, d_low_en, num_st * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_Jh);
    cudaFree(d_energies);
    cudaFree(d_low_en);
}


template
void find_lowest<float>(
    float* Jh,
    int N,
    int chunk_exp,
    float* en_out,
    long int* st_out,
    int num_st,
    int grid_size,
    int block_size,
    void* user_data,
    callback_function callback
);

template
void find_lowest<double>(
    double* Jh,
    int N,
    int chunk_exp,
    double* en_out,
    long int* st_out,
    int num_st,
    int grid_size,
    int block_size,
    void* user_data,
    callback_function callback
);


template
void find_lowest_energies_only<float>(
    float* Jh,
    int N,
    int chunk_exp,
    float* en_out,
    int num_st,
    int grid_size,
    int block_size,
    void* user_data,
    callback_function callback
);

template
void find_lowest_energies_only<double>(
    double* Jh,
    int N,
    int chunk_exp,
    double* en_out,
    int num_st,
    int grid_size,
    int block_size,
    void* user_data,
    callback_function callback
);

void getGPUMemInfo(unsigned long* free, unsigned long* total)
{
    cudaMemGetInfo(free, total);
}
