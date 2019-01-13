module bucketselect
  interface top_k
     subroutine top_k_double(d_vector, length, k) bind(C, name="top_k_double")
       use iso_c_binding
       real(c_double) :: d_vector(*)
       integer(c_int), value :: length
       integer(c_int), value :: k
     end subroutine top_k_double
  end interface top_k

  interface top_k_by_key
     subroutine top_k_int_by_key(values, keys, length, k) bind (C, name="top_k_int_by_key")
       use iso_c_binding
       integer(c_int64_t) ::  values(*)
       real(c_double) :: keys(*)       
       integer(c_int), value :: length
       integer(c_int), value :: k
     end subroutine top_k_int_by_key
  end interface top_k_by_key
end module bucketselect
