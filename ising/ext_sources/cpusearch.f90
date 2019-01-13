module types
  integer, parameter :: ik = 8
  integer(ik), parameter :: d = 2
  integer, parameter :: wp = kind(1.0D0)
end module types

module cpusearch
implicit none
contains

  subroutine find_lowest(& ! Find states of lowest energies
       Jh, &               ! Coefficients of hamiltonian
       no_bits, &          ! Number of bits
       sweep_size, &       ! 2 ** sweep_size states will be searched at once
       energies_out, &     ! Array to store lowest energies in
       states_out, &       ! Array to store states correspondingx to lo_energies
       how_many, &         ! How many energies to keep track of?
       omp_threads)        ! how many OMP threads to use

    use omp_lib
    use types
    use bucketselect
    use thrust
    
    !f2py intent(callback) callback
    external :: callback
    
    integer(ik), intent(inout) :: how_many
    real(wp), intent(in) :: Jh(no_bits, no_bits)
    integer, intent(in) :: omp_threads, sweep_size, no_bits
    real(wp), intent(out) :: energies_out(how_many)
    integer(ik), intent(out) :: states_out(how_many)
    real(wp), allocatable :: lowest(:)
    integer(ik), allocatable :: lowest_states(:)
    real(wp), allocatable :: energies(:)
    integer(ik), allocatable :: states(:)
    integer(ik) :: m, k, info, state_repr

    allocate(energies(d ** sweep_size), states(d ** sweep_size))

    if (how_many > 2 ** sweep_size) then
       how_many = 2 ** sweep_size
    end if

    allocate(lowest(how_many * 2), lowest_states(how_many * 2) )

    do m = 1_ik, d ** (no_bits - sweep_size)
       !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(state_repr)
       do k = 1_ik, d ** (sweep_size)
          state_repr = k + ((m - 1_ik) * (d ** sweep_size))
          energies(k) = energy(state_repr, Jh, no_bits)
          states(k) = state_repr
       end do
       !$OMP END PARALLEL DO
       call top_k_int_by_key(states, energies, int(d ** sweep_size, kind=4), int(d** sweep_size-how_many+1, kind=4))

       if (m == 1) then
          lowest(1:how_many) = energies(1:how_many)
          lowest_states(1:how_many) = states(1:how_many)
       else
          lowest(how_many+1: 2 * how_many) = energies(1:how_many)
          lowest_states(how_many+1: 2 * how_many) = states(1:how_many)
          call top_k_int_by_key(lowest_states, lowest, int(2 * how_many, kind=4), int(how_many+1, kind=4))
       end if
       call callback(m)
    end do

    call sort_by_key_double(lowest, int(how_many, kind=4), lowest_states)
    energies_out(1:how_many) = lowest(1:how_many)
    states_out(1:how_many) = lowest_states(1:how_many)

    deallocate(lowest)
    deallocate(lowest_states)
    deallocate(energies)
    deallocate(states)
  end subroutine find_lowest

  subroutine find_lowest_energies_only(& ! Find states of lowest energies
       Jh, &               ! Coefficients of hamiltonian
       no_bits, &          ! Number of bits
       sweep_size, &       ! 2 ** sweep_size states will be searched at once
       energies_out, &     ! Array to store lowest energies in
       how_many, &         ! How many energies to keep track of?
       omp_threads)        ! how many OMP threads to use

    use omp_lib
    use types
    use bucketselect
    use thrust
    
    !f2py intent(callback) callback
    external :: callback
    
    integer(ik), intent(inout) :: how_many
    real(wp), intent(in) :: Jh(no_bits, no_bits)
    integer, intent(in) :: omp_threads, sweep_size, no_bits
    real(wp), intent(out) :: energies_out(how_many)
    real(wp), allocatable :: lowest(:)
    real(wp), allocatable :: energies(:)
    integer(ik) :: m, k, info, state_repr

    allocate(energies(d ** sweep_size))

    if (how_many > 2 ** sweep_size) then
       how_many = 2 ** sweep_size
    end if

    allocate(lowest(how_many * 2))

    do m = 1_ik, d ** (no_bits - sweep_size)
       !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(state_repr)
       do k = 1_ik, d ** (sweep_size)
          state_repr = k + ((m - 1_ik) * (d ** sweep_size))
          energies(k) = energy(state_repr, Jh, no_bits)
       end do
       !$OMP END PARALLEL DO
       call top_k(energies, int(d ** sweep_size, kind=4), int(d** sweep_size-how_many+1, kind=4))

       if (m == 1) then
          lowest(1:how_many) = energies(1:how_many)
       else
          lowest(how_many+1: 2 * how_many) = energies(1:how_many)
          call top_k(lowest, int(2 * how_many, kind=4), int(how_many+1, kind=4))
       end if
       call callback(m)
    end do

    call thrust_sort(lowest, int(how_many, kind=4))
    energies_out(1:how_many) = lowest(1:how_many)

    deallocate(lowest)
    deallocate(energies)
  end subroutine find_lowest_energies_only

  real(wp) pure function energy(state_repr, Jh, no_bits)
    use types
    integer :: i, j
    integer :: state(64)
    integer(ik), intent(in) :: state_repr
    real(8), intent(in) :: Jh(:,:)
    integer, intent(in) :: no_bits
    state = 0

    do i = 0, no_bits - 1
       if (btest(state_repr, i)) state(no_bits-i) = 1
    end do

    energy = 0.0D0

    do i = 1, no_bits
       if (state(i) == 1) then
          energy = energy - sum(Jh(i, i+1:no_bits) * state(i+1:no_bits) * state(i)) - Jh(i, i) * state(i)
       end if
    end do

  end function energy

end module cpusearch
