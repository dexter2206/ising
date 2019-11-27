/* Copyright 2011 Russel Steinbach, Jeffrey Blanchard, Bradley Gordon,
 *   and Toluwaloju Alabi
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include "select.h"
using namespace std;


#define CUTOFF_POINT 2200000 


  template<typename T>
  void cleanup(uint *h_c, T* d_k, int *etb, uint *bc){
    free(h_c);
    free(d_k);
    free(etb);
    free(bc);
  }

//this function assigns elements to buckets
template <typename T>
void assignBucket(T* d_vector, int length, int bucketNumbers, double slope, double minimum, int* bucket, uint* bucketCount){

    int i;
      
#pragma omp parallel 
  {
    int bucketIndex;
    int* thread_bucket_counts;
    thread_bucket_counts = NULL;
    thread_bucket_counts = (int *) calloc(bucketNumbers, sizeof(int));

  #pragma omp  for 
  for(i=0; i  < length; i++) {
    bucketIndex =  (d_vector[i] - minimum) * slope;
    if(bucketIndex >= bucketNumbers) {
      bucketIndex = bucketNumbers - 1;
    }
    bucket[i] = bucketIndex;
    thread_bucket_counts[bucketIndex]++;
  }

  
#pragma omp barrier
 
    for(bucketIndex = 0; bucketIndex < bucketNumbers; bucketIndex++) {
#pragma omp atomic
      bucketCount[bucketIndex] += thread_bucket_counts[bucketIndex];
    }
    free(thread_bucket_counts);
  }
}

//this function reassigns elements to buckets
template <typename T>
void reassignBucket(T* d_vector, int *bucket, uint *bucketCount, const int bucketNumbers, const int length, const double slope, const double maximum, const double minimum, int Kbucket){
  int i;  
#pragma omp parallel 
  {  

  int bucketIndex;
  int * thread_bucket_counts = NULL;
  thread_bucket_counts = (int *) calloc(bucketNumbers, sizeof(int));
  
#pragma omp for
  for(i=0; i < length; i++) {
    if(bucket[i] != Kbucket){
      bucket[i] = bucketNumbers+1;
    }
    else{
      //calculate the bucketIndex for each element
      bucketIndex = (d_vector[i] - minimum) * slope;

      //if it goes beyond the number of buckets, put it in the last bucket
      if(bucketIndex >= bucketNumbers){
	bucketIndex = bucketNumbers - 1;
      }
      bucket[i] = bucketIndex;
      thread_bucket_counts[bucketIndex]++;
    }
  }

#pragma omp barrier
 
    for(bucketIndex = 0; bucketIndex < bucketNumbers; bucketIndex++) {
#pragma omp atomic
      bucketCount[bucketIndex] += thread_bucket_counts[bucketIndex];
    }
    free(thread_bucket_counts);
  }
}

//copy elements in the kth bucket to a new array
template <typename T>
void copyElement(T* d_vector, int length, int* elementArray, int bucket, T* newArray, uint* count){
  int i;
#pragma omp parallel for  private(i)
  for(i=0; i < length; i++) {
    if(elementArray[i] == bucket) {
      #pragma omp critical
      {
	newArray[count[0]++] = d_vector[i];
      }
    }
  }
}

//this function finds the bin containing the kth element we are looking for (works on the host)
  inline int FindKBucket(uint *d_counter, uint *h_counter, const int num_buckets, const int k, uint * sum){
    memcpy(sum, d_counter, sizeof(uint));
    int Kbucket = 0;
    if (sum[0]<k){
      memcpy(h_counter, d_counter, num_buckets * sizeof(uint));
      while ( (sum[0]<k) & (Kbucket<num_buckets-1)){
        Kbucket++; 
        sum[0] = sum[0] + h_counter[Kbucket];
      }
    }
    else{
      memcpy(h_counter, d_counter, sizeof(uint));
    }
  
    return Kbucket;
  }

template <typename T>
void GetKvalue(T* d_vector, int * d_bucket, const int Kbucket, const int n, T* Kvalue)
{
  int i;
#pragma omp parallel for private(i)
  for(i=0; i < n; i++) {
    if ( d_bucket[i] == Kbucket ) {
      Kvalue[0] = d_vector[i];
    }    
  }
}


/************************************************************************/
/************************************************************************/
//THIS IS THE PHASE TWO FUNCTION WHICH WILL BE CALLED IF THE INPUT
//LENGTH IS LESS THAN THE CUTOFF OF 2MILLION 200 THOUSAND
/************************************************************************/


template <typename T>
T phaseTwo(T* d_vector, int length, int K, double maxValue = 0, double minValue = 0){
  //  cout << "Phase two called" <<  " length " << length << " k " << K << endl;
  uint sum=0, Kbucket=0, iter=0;
  int Kbucket_count = 0;
  int numBuckets = 1024; 
  //variable to store the end result
  T kthValue =0;

  //declaring and initializing other variables
  size_t size = length * sizeof(int);
  size_t totalBucketSize = numBuckets * sizeof(uint);

  //allocate memory to store bucket assignments and to count elements in buckets
  int* elementToBucket;
  uint* d_bucketCount;
  elementToBucket = (int *) malloc(size);
  d_bucketCount = (uint *) calloc(numBuckets, sizeof(uint));
  uint * h_bucketCount = (uint*) malloc(totalBucketSize);

  T* d_Kth_val;
  d_Kth_val = (T*) malloc(sizeof(T));

  //if max == min, then we know that it must not have had the values passed in. 
  if(maxValue == minValue){
    thrust::pair<T*, T*> result = thrust::minmax_element(d_vector, d_vector + length);
    minValue = *result.first;
    maxValue = *result.second;
  }
  
  double slope = (numBuckets - 1)/(maxValue - minValue);
  //first check is max is equal to min
  if(maxValue == minValue){
    cleanup(h_bucketCount, d_Kth_val, elementToBucket,d_bucketCount);
    return maxValue;
  }

  //make all entries of this vector equal to zero
  memset(d_bucketCount, 0, sizeof(uint) * numBuckets);

  //distribute elements to bucket
  //  cout << "before assign buckets " << endl;
  assignBucket(d_vector, length, numBuckets, slope, minValue, elementToBucket, d_bucketCount);

  //find the bucket containing the kth element we want
  //  cout << "before findkbucket" << endl;
  Kbucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &sum);

  Kbucket_count = h_bucketCount[Kbucket];
  
  //  cout << "Before while loop " << endl;
  while ( (Kbucket_count > 1) && (iter < 1000)){
    minValue = max(minValue, minValue + Kbucket/slope);
    maxValue = min(maxValue, minValue + 1/slope);
    K = K - sum + Kbucket_count;
    if ( maxValue - minValue > 0.0f ){
      slope = (numBuckets - 1)/(maxValue-minValue);
      memset(d_bucketCount, 0, sizeof(uint) * numBuckets);
      reassignBucket(d_vector, elementToBucket, d_bucketCount, numBuckets,length, slope, maxValue, minValue, Kbucket);

      sum = 0;
      Kbucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, &sum);
      Kbucket_count = h_bucketCount[Kbucket];
      iter++;
    }
    else{
      //if the max and min are the same, then we are done
      cleanup(h_bucketCount, d_Kth_val, elementToBucket, d_bucketCount);
      return maxValue;
    }
  }

  //  cout << "After while loop " << endl;
    GetKvalue(d_vector, elementToBucket, Kbucket, length, d_Kth_val);
    //    cout << "After getkvalue " << endl;
    memcpy(&kthValue, d_Kth_val, sizeof(T));
    //    cout << "After memcpy " << endl;
  cleanup(h_bucketCount, d_Kth_val, elementToBucket, d_bucketCount);
  //  cout << " After cleanup " << endl;
  return kthValue;
}



/* this function finds the kth-largest element from the input array */
template <typename T>
T phaseOne(T* d_vector, int length, int K, int pass = 0){
  //  cout << "Phase one called" <<  " length " << length << " k " << K << endl;
  //  cout << "We will return to " <<  __builtin_return_address(0) << endl;
  //declaring variables for kernel launches
  int kthBucket, kthBucketCount;
  int newInputLength;
  int numBuckets = 1024;  
  int* elementToBucket; //array showing what bucket every element is in
  //declaring and initializing other variables

  uint *d_bucketCount, *count; //array showing the number of elements in each bucket
  uint kthBucketScanSize = 0;

  size_t size = length * sizeof(int);

  //variable to store the end result
  T kthValue = 0;
  T* newInput;

  //find max and min with thrust
  double maximum, minimum;

  thrust::pair<T*, T*> result = thrust::minmax_element(d_vector, d_vector + length);

  minimum = *result.first;
  maximum = *result.second;

  //if the max and the min are the same, then we are done
  if(maximum == minimum){
    return maximum;
  }
  //if we want the max or min just return it
  if(K == 1){
    return minimum;
  }
  if(K == length){
    return maximum;

  }		
  //Allocate memory to store bucket assignments
  
  elementToBucket = (int *) malloc(size);

  //Allocate memory to store bucket counts
  size_t totalBucketSize = numBuckets * sizeof(uint);
  d_bucketCount = (uint *)malloc(totalBucketSize);
  uint* h_bucketCount = (uint*)malloc(totalBucketSize);

  //Calculate max-min
  double range = maximum - minimum;
  //Calculate the slope, i.e numBuckets/range
  double slope = (numBuckets - 1)/range;

  count = (uint *)malloc(sizeof(uint));
  //Set the bucket count vector to all zeros
  memset(d_bucketCount, 0, sizeof(uint) * numBuckets);

  //Distribute elements into their respective buckets
  assignBucket(d_vector, length, numBuckets, slope, minimum, elementToBucket, d_bucketCount);
  kthBucket = FindKBucket(d_bucketCount, h_bucketCount, numBuckets, K, & kthBucketScanSize);
  kthBucketCount = h_bucketCount[kthBucket];
 

  //we must update K since we have reduced the problem size to elements in the kth bucket
  if(kthBucket != 0){
    K = kthBucketCount - (kthBucketScanSize - K);
  }

  //copy elements in the kth bucket to a new array
  newInput = (T *) malloc(kthBucketCount * sizeof(T));
  memset(count, 0, sizeof(uint));
  copyElement(d_vector, length, elementToBucket, kthBucket, newInput, count);



  //store the length of the newly copied elements
  newInputLength = kthBucketCount;

  int zeros = 0;
  for(int j = 0; j < newInputLength; j++) {
    if(newInput[j] == 0) {
      zeros++;
    }
  }

  //if we only copied one element, then we are done
  if(newInputLength == 1){
    kthValue = newInput[0];
      
    //free all used memory
    free(elementToBucket);
    free(d_bucketCount);
    free(h_bucketCount);
    free(count);
    free(newInput);
    return kthValue;
  }
 
  /*********************************************************************/
  //END OF FIRST PASS, NOW WE PROCEED TO SUBSEQUENT PASSES
  /*********************************************************************/

  //if the new length is greater than the CUTOFF, run the regular phaseOne again
  if(newInputLength > CUTOFF_POINT && pass < 1){
    if(pass > 0){
      free(d_vector);
    }
    // free(elementToBucket);
    // free(d_bucketCount); free(count);
    kthValue = phaseOne(newInput, newInputLength, K, pass + 1);
    //    cout << "After phase one again..." << endl;
  }
  else{
    minimum = max(minimum, minimum + kthBucket/slope);
    maximum = min(maximum, minimum + 1/slope);
    kthValue = phaseTwo(newInput,newInputLength, K, maximum, minimum);
  }

  //free all used memory
  //  cout << "Freeing elementToBucket" << endl;
  free(elementToBucket);
  //  cout << "Freeing d_bucket " << endl;  
  free(d_bucketCount);
  //  cout << "Freeing h_bucket " << endl;    
  free(h_bucketCount);
  //  cout << "Freeing newInput " << endl;    
  free(newInput);
  //  cout << "Freeing count " << endl;      
  free(count);
  //  cout << "Everything freed " << kthValue << endl;
  return kthValue;
}

/**************************************************************************/
/**************************************************************************/
//THIS IS THE BUCKETSELECT FUNCTION WRAPPER THAT CHOOSES THE CORRECT VERSION
//OF BUCKET SELECT TO RUN BASED ON THE INPUT LENGTH
/**************************************************************************/
template <typename T>
T bucketSelectWrapper(T* d_vector, int length, int K)
{
  T kthValue;
  //change K to be the kth smallest
  K = length - K + 1;

  if(length <= CUTOFF_POINT)
    {
      kthValue = phaseTwo(d_vector, length, K);
      return kthValue;
    }
  else
    {
      kthValue = phaseOne(d_vector, length, K);    
      return kthValue;
    }
}


template <typename T>
struct lt
{
  T pivot;
  
  lt(T pivot)
  {
    this->pivot = pivot;
  }

  __host__
  bool operator()(const T &x)
  {
    return x < pivot;
  }
};

template <typename T>
struct lt_by_key
{
  T pivot;

  lt_by_key(T pivot)
  {
    this->pivot = pivot;
  }

  __host__
  bool operator()(const thrust::tuple<int64_t, T> &x)
  {
    return thrust::get<1>(x) < pivot;
  }
};

template <typename T>
struct le
{
  T pivot;
  
  le(T pivot)
  {
    this->pivot = pivot;
  }

  __host__
  bool operator()(const T &x)
  {
    return x <= pivot;
  }
};

template <typename T>
struct le_by_key
{
  T pivot;

  le_by_key(T pivot)
  {
    this->pivot = pivot;
  }

  __host__
  bool operator()(const thrust::tuple<int64_t, T> &x)
  {
    return thrust::get<1>(x) <= pivot;
  }
};


template <typename T>
void partition(T *input, int64_t length, T pivot)
{
  le<T> predicate_le(pivot);
  lt<T> predicate_lt(pivot);
  T* middle = thrust::partition(input, input+length, predicate_lt);
  thrust::partition(middle, input+length, predicate_le);
}

template <typename T>
int64_t partition_int_by_key(int64_t *values, T *keys, int64_t length, T pivot)
{
  le_by_key<T> predicate_le(pivot);
  lt_by_key<T> predicate_lt(pivot);  
  thrust::zip_iterator<thrust::tuple<int64_t*, T*> > start = thrust::make_zip_iterator(thrust::make_tuple(values, keys));
  thrust::zip_iterator<thrust::tuple<int64_t*, T*> > middle = thrust::partition(start, start+length, predicate_lt);
  thrust::partition(middle, start+length, predicate_le);
}

template <typename T>
void top_k(T* vector, int length, int k)
{
  T pivot =  bucketSelectWrapper(vector, length, k);
  partition(vector, length, pivot);
}


template <typename T>
void top_k_int_by_key(int64_t* values, T* keys, int length, int k) {
  T pivot =  bucketSelectWrapper<T>(keys, length, k);
  partition_int_by_key(values, keys, length, pivot);
}


template void top_k<float>(float* vector, int length, int k);
template void top_k<double>(double* vector, int length, int k);

template void top_k_int_by_key<float>(int64_t* values, float* keys, int length, int k);
template void top_k_int_by_key<double>(int64_t* values, double* keys, int length, int k);
