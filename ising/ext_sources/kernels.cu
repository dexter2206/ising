#include <stdio.h>

template <typename T>
__device__
T energy(int64_t state_repr, T* Q, int N)
{
  int i, j;
  short int bit;
  T energy = 0;
  
  for(i = 0; i < N; i++) {
    bit = (state_repr >> i) & 1;
    if(bit) {
      for(j = i; j < N; j++) {
	energy -= Q[i * N + j] * bit * ((state_repr >> j) & 1);
      }
    }
  }
  return energy;
}


template <typename T>
void __global__ search(T* Q, int N, int sweep_size, T* energies, long int* states, long int m)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int i;
  long int state_repr;
  if(idx < sweep_size) {
    state_repr = idx + m;
    states[idx] = state_repr;
    energies[idx] = energy(state_repr, Q, N);
  }
}


template <typename T>
void __global__ search_energies_only(T* Q, int N, int sweep_size, T* energies, long int m)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int i;
  long int state_repr;
  if(idx < sweep_size) {
    state_repr = idx + m;
    energies[idx] = energy(state_repr, Q, N);
  }    
}


template
__global__
void search<float>(float* Q, int N, int sweep_size, float* energies, long int* states, long int m);


template
__global__
void search<double>(double* Q, int N, int sweep_size, double* energies, long int* states, long int m);


template
__global__
void search_energies_only<float>(float* Q, int N, int sweep_size, float* energies, long int m);


template
__global__
void search_energies_only<double>(double* Q, int N, int sweep_size, double* energies, long int m);
