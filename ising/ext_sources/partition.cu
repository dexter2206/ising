#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>

extern "C" {
  struct lt
  {
    double pivot;

    lt(double pivot)
    {
      this->pivot = pivot;
    }

    __host__
    bool operator()(const double &x)
    {
      return x < pivot;
    }
  };

  struct lt_by_key
  {
    double pivot;

    lt_by_key(double pivot)
    {
      this->pivot = pivot;
    }

    __host__
    bool operator()(const thrust::tuple<int64_t, double> &x)
    {
      return thrust::get<1>(x) < pivot;
    }
  };

  int64_t partition_double(double *input, int64_t length, double pivot)
  {
    thrust::device_ptr<double> data(input);
    lt predicate(pivot);
    return thrust::partition(input, input+length, predicate) - input;
  }

  int64_t partition_int_by_key(int64_t *values, double *keys, int64_t length, double pivot)
  {
    lt_by_key predicate(pivot);
    thrust::zip_iterator<thrust::tuple<int64_t*, double*> > start = thrust::make_zip_iterator(thrust::make_tuple(values, keys));
    return thrust::partition(start, start+length, predicate) - start;
  }
}
