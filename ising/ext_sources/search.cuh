template <typename T>
void test(T x);

typedef int (*callback_function)(int, void*);

template <typename T>
void find_lowest(
  T* Jh,
  int num_bits,
  int chunk_exponent,
  T* energies_out,
  long int* states_out,
  int num_states,
  int grid_size,
  int block_size,
  void* user_data,
  callback_function callback
);

template <typename T>
void find_lowest_energies_only(
  T* Jh,
  int num_bits,
  int chunk_exponent,
  T* energies_out,
  int num_states,
  int grid_size,
  int block_size,
  void *,
  callback_function callback
);

void getGPUMemInfo(unsigned long* free, unsigned long* total);