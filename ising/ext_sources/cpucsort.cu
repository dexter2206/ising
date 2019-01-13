// Filename: csort.cu
// nvcc -c -arch sm_xx csort.cu

#include <thrust/device_vector.h>
#include <thrust/sort.h>

extern "C" {

//Sort for integer arrays
void sort_int_wrapper( int *data, int N)
{
// Wrap raw pointer with a device_ptr
thrust::device_ptr <int> dev_ptr(data);

// Use device_ptr in Thrust sort algorithm
thrust::sort(dev_ptr, dev_ptr+N);
}
//Sort for float arrays
void sort_float_wrapper( float *data, int N)
{
thrust::device_ptr <float> dev_ptr(data);
thrust::sort(dev_ptr, dev_ptr+N);
}
//Sort for double arrays
void sort_double_wrapper( double *data, int N)
{
thrust::device_ptr <double> dev_ptr(data);
thrust::sort(data, data+N);
}
//Sort by key for double arrays
void sort_by_key_float_wrapper(float *keys, int N, int64_t *values)
{
thrust::device_ptr <float> dev_ptr(keys);
thrust::device_ptr <int64_t> dev_ptr_2(values);
thrust::sort_by_key(keys, keys+N, values);
}
void sort_by_key_double_wrapper(double *keys, int N, int64_t *values)
{
thrust::device_ptr <double> dev_ptr(keys);
thrust::device_ptr <int64_t> dev_ptr_2(values);
thrust::sort_by_key(keys, keys+N, values);
}
}
