module thrust
!
interface thrust_sort
!
subroutine sort_int(input, N) bind(C, name="sort_int_wrapper")
use iso_c_binding
integer(c_int) :: input(*)
integer(c_int), value :: N
end subroutine
!
subroutine sort_float(input, N) bind(C, name="sort_float_wrapper")
use iso_c_binding
real(c_float) :: input(*)
integer(c_int), value :: N
end subroutine
!
subroutine sort_double(input, N) bind(C, name="sort_double_wrapper")
use iso_c_binding
real(c_double) :: input(*)
integer(c_int), value :: N
end subroutine
!
end interface
!
interface thrust_sort_by_key
!
subroutine sort_by_key_double(keys, N, values) bind(C, name="sort_by_key_double_wrapper")
use iso_c_binding
real(c_double) :: keys(*)
integer(c_int64_t) :: values(*)
integer(c_int), value :: N
end subroutine
!
subroutine sort_by_key_float(keys, N, values) bind(C, name="sort_by_key_float_wrapper")
use iso_c_binding
real(c_float) :: keys(*)
integer(c_int64_t) :: values(*)
integer(c_int), value :: N
end subroutine
!
!
end interface
!
end module thrust
