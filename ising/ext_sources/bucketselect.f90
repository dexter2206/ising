module bucketselect
  interface top_k
     subroutine top_k_double(d_vector, length, k, num_blocks, num_threads) bind(C, name="top_k_double")
       use iso_c_binding
       real(c_double), device :: d_vector(*)
       integer(c_int), value :: length
       integer(c_int), value :: k
       integer(c_int), value :: num_blocks
       integer(c_int), value :: num_threads
    end interface top_k
    
  interface top_k_by_key
     subroutine top_k_int_by_key(values, keys, length, k, num_blocks, num_threads) bind (C, name="top_k_int_by_key")
       use iso_c_binding
       integer(c_int64_t), device ::  values(*)
       real(c_double), device :: keys(*)       
       integer(c_int), value :: length
       integer(c_int), value :: k
       integer(c_int), value :: num_blocks
       integer(c_int), value :: num_threads       
     end subroutine top_k_int_by_key
  end interface top_k_by_key    
end module bucketselect
