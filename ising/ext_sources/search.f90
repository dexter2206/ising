module cuda_search
!
  use cudafor
  use global, only: wp, ik
!
implicit none

!
contains
!
   attributes(global) subroutine search(Jh_d, no_bits, sweep_size, energies, states, m)
   real(wp) :: en
   integer :: i, j
   real(wp), intent(in), device :: Jh_d(:,:)
   integer(ik) :: idx, state_repr
   integer, device :: state(8 * ik)
   integer(ik), intent(in), value :: m, no_bits, sweep_size
   real(wp), intent(out), device :: energies(*)
   integer(ik), intent(out), device :: states(*)

!
	 idx = threadIdx%x + blockDim%x * (blockIdx%x-1)
!
	 if (idx <= 2 ** sweep_size) then
!
             state_repr = idx + m
             states(idx) = state_repr
!
             state = 0
             do i = 0, no_bits-1
                if (btest(state_repr, i)) state(no_bits-i) = 1
             end do
!
             en = 0.0_wp
             do i = 1, no_bits
                if (state(i) == 1) then
                   en = en - sum(Jh_d(i, i+1:no_bits) * state(i+1:no_bits) * state(i)) - Jh_d(i, i) * state(i)
                end if
             end do
             energies(idx) = en
         end if
         return
         
 end subroutine search

 attributes(global) subroutine search_energies_only(Jh_d, no_bits, sweep_size, energies, m)
   real(wp) :: en
   integer :: i, j
   real(wp), intent(in), device :: Jh_d(:,:)
   integer(ik) :: idx, state_repr
   integer, device :: state(8 * ik)
   integer(ik), intent(in), value :: m, no_bits, sweep_size
   real(wp), intent(out), device :: energies(*)

!
	 idx = threadIdx%x + blockDim%x * (blockIdx%x-1)
!
	 if (idx <= 2 ** sweep_size) then
!
             state_repr = idx + m
!
             state = 0
             do i = 0, no_bits-1
                if (btest(state_repr, i)) state(no_bits-i) = 1
             end do
!
             en = 0.0_wp
             do i = 1, no_bits
                if (state(i) == 1) then
                   en = en - sum(Jh_d(i, i+1:no_bits) * state(i+1:no_bits) * state(i)) - Jh_d(i, i) * state(i)
                end if
             end do
             energies(idx) = en
         end if
   return
 end subroutine search_energies_only
 
!
end module cuda_search
