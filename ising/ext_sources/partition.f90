module partition
implicit none
  interface swap
     module procedure swap_real
     module procedure swap_int
  end interface swap
contains
  subroutine swap_real(first, second)
    real(8), intent(inout) :: first, second
    real(8) :: tmp
    tmp = first
    first = second
    second = tmp
  end subroutine swap_real

  subroutine swap_int(first, second)
    integer(8), intent(inout) :: first, second
    integer(8) :: tmp
    tmp = first
    first = second
    second = tmp
  end subroutine swap_int

  subroutine partition_it(input, length, pivot, pivot_pos)
    use thrust
    integer(8), intent(in) :: length
    real(8), intent(inout) :: input(length)
    real(8), intent(in) :: pivot
    integer(8), intent(out) :: pivot_pos
    pivot_pos = thrust_partition(input, length, pivot)
  end subroutine partition_it

  subroutine quickselect(input, length, k)
    use thrust
    integer(8), intent(in) :: length
    real(8), intent(inout) :: input(length)
    integer(8), intent(in) :: k
    integer(8) :: left, right, pivot_pos
    
    left = 1
    right = length

    if (k >= length) then
       return
    end if
    pivot_pos = -1

    do while (pivot_pos .ne. k)
       pivot_pos = left + thrust_partition(input(left:), right-left, input(right))
       call swap(input(pivot_pos), input(right))

       if (pivot_pos < k) then
          left = pivot_pos + 1
       end if
       if (pivot_pos > k) then
          right = pivot_pos - 1
       end if
    end do
  end subroutine quickselect

  subroutine quickselect_by_key(values, keys, length, k)
    use thrust
    integer(8), intent(in) :: length
    real(8), intent(inout) :: keys(length)
    integer(8), intent(inout) :: values(length)
    integer(8), intent(in) :: k
    integer(8) :: left, right, pivot_pos
    real(8) :: start, stop;
    
    left = 1
    right = length
    pivot_pos = -1

    if (k >= length) then
       return
    end if
    do while (pivot_pos .ne. k)
       pivot_pos = left + thrust_partition_by_key(values(left:), keys(left:), right-left, keys(right))
       call swap(keys(pivot_pos), keys(right))
       call swap(values(pivot_pos), values(right))

       if (pivot_pos < k) then
          left = pivot_pos + 1
       end if
       if (pivot_pos > k) then
          right = pivot_pos - 1
       end if
    end do
  end subroutine quickselect_by_key
end module partition
