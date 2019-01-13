module thrust
  implicit none
  interface thrust_partition
     integer(c_int64_t) function partition_double(input, length, pivot) bind(C, name="partition_double")
       use iso_c_binding
       real(c_double)  :: input(*)
       integer(c_int64_t), value :: length
       real(c_double), value :: pivot
     end function partition_double
  end interface thrust_partition

  interface thrust_partition_by_key
     integer(c_int64_t) function partition_int_by_key(values, keys, length, pivot) bind(C, name="partition_int_by_key")
       use iso_c_binding
       real(c_double) :: keys(*)
       integer(c_int64_t) :: values(*)
       integer(c_int64_t), value :: length
       real(c_double), value:: pivot
     end function partition_int_by_key
  end interface thrust_partition_by_key
end module thrust
