module gpusearch
implicit none

contains

  subroutine find_lowest(& ! Find states of lowest energies
       Jh, &               ! Coefficients of hamiltonian
       no_bits, &          ! Number of bits
       sweep_size, &       ! 2 ** sweep_size states will be searched at once
       energies_out, &     ! Array to store lowest energies in
       states_out, &       ! Array to store states correspondingx to lo_energies
       how_many)           ! How many energies to keep track of?

    use thrust
    use cudafor
    use global, only : wp, ik, d
    use cuda_search, only : search
    use bucketselect
    type(dim3) :: grid, tBlock
    integer(ik), intent(in) :: how_many, no_bits, sweep_size
    real(wp), intent(out) :: energies_out(how_many)
    integer(ik), intent(out) :: states_out(how_many)
    real(wp), intent(in) :: Jh(no_bits, no_bits)
    real(wp), allocatable, device :: lowest_d(:), energies_d(:)
    integer(ik), allocatable, device :: lowest_states_d(:), states_d(:)
    real(wp), allocatable, device :: Jh_d(:,:)
    integer(ik) :: idx, m, istat

    !f2py intent(callback) callback
    external :: callback

    real(wp), allocatable :: tmp_energies(:)

    tBlock = dim3(32, 1, 1)
    grid = dim3(ceiling(real(d**sweep_size)/tBlock%x), 1, 1)

    allocate(Jh_d(no_bits, no_bits))
    allocate(lowest_d(2 * how_many), lowest_states_d(2 * how_many))
    allocate(states_d(d**sweep_size), energies_d(d**sweep_size))

    Jh_d = Jh
    Jh = Jh_d

    do m = 1_ik, d ** (no_bits - sweep_size)
       idx = (m-1_ik) * d**sweep_size - 1_ik
       call search<<<grid,tBlock>>>(Jh_d, no_bits, sweep_size, energies_d, states_d, idx)

       call top_k_int_by_key(states_d, energies_d, int(d ** sweep_size, kind=4), int(d** sweep_size-how_many+1, kind=4), int(40, kind=4), int(1024, kind=4))

       if (m == 1) then
          istat = cudaMemcpy(lowest_d, energies_d, int(how_many, kind=4))
          istat = cudaMemcpy(lowest_states_d, states_d, int(how_many, kind=4))
       else
          istat = cudaMemcpy(lowest_d(how_many+1: 2 * how_many), energies_d, int(how_many, kind=4))
          istat = cudaMemcpy(lowest_states_d(how_many+1: 2 * how_many), states_d, int(how_many, kind=4))!          
          call top_k_int_by_key(lowest_states_d, lowest_d, int(2 * how_many, kind=4), int(how_many+1, kind=4),int(40, kind=4), int(1024, kind=4))
          
       end if
       call callback(m)
    end do
    call thrust_sort_by_key(lowest_d, int(how_many, kind=4), lowest_states_d)
    states_out(1:how_many) = lowest_states_d(1:how_many)
    energies_out(1:how_many) = lowest_d(1:how_many)

    deallocate(lowest_d)
    deallocate(lowest_states_d)
    deallocate(energies_d)
    deallocate(states_d)
    deallocate(Jh_d)
  end subroutine find_lowest

  subroutine find_lowest_energies_only(& ! Find states of lowest energies
       Jh, &                             ! Coefficients of hamiltonian
       no_bits, &                        ! Number of bits
       sweep_size, &                     ! 2 ** sweep_size states will be searched at once
       energies_out, &                   ! Array to store lowest energies in
       how_many)                         ! How many energies to keep track of?

    use bucketselect
    use cudafor
    use global, only : wp, ik, d
    use cuda_search, only : search_energies_only
    use thrust;
    type(dim3) :: grid, tBlock
    integer(ik), intent(in) :: how_many, no_bits, sweep_size
    real(wp), intent(out) :: energies_out(how_many)
    real(wp), intent(in) :: Jh(no_bits, no_bits)
    real(wp), allocatable, device :: lowest_d(:), energies_d(:)
    real(wp), allocatable, device :: Jh_d(:,:)
    integer(ik) :: idx, m, istat

    !f2py intent(callback) callback
    external :: callback

    real(wp), allocatable :: tmp_energies(:)

    tBlock = dim3(32, 1, 1)
    grid = dim3(ceiling(real(d**sweep_size)/tBlock%x), 1, 1)

    allocate(Jh_d(no_bits, no_bits))
    allocate(lowest_d(2 * how_many))
    allocate(energies_d(d**sweep_size))


    Jh_d = Jh
    Jh = Jh_d

    do m = 1_ik, d ** (no_bits - sweep_size)
       idx = (m-1_ik) * d**sweep_size - 1_ik

       call search_energies_only<<<grid,tBlock>>>(Jh_d, no_bits, sweep_size, energies_d, idx)

       call top_k_double(energies_d, int(d ** sweep_size, kind=4), int(d** sweep_size-how_many+1, kind=4), int(40, kind=4), int(1024, kind=4))

       if (m == 1) then
          istat = cudaMemcpy(lowest_d, energies_d, int(how_many, kind=4))
       else
          istat = cudaMemcpy(lowest_d(how_many+1: 2 * how_many), energies_d, int(how_many, kind=4))
       call top_k_double(lowest_d, int(2 * how_many, kind=4), int(how_many+1, kind=4), int(40, kind=4), int(1024, kind=4))
       end if
       call callback(m)
    end do
    
    call thrust_sort(lowest_d, int(how_many, kind=4))
    energies_out(1:how_many) = lowest_d(1:how_many)

    deallocate(lowest_d)
    deallocate(energies_d)
    deallocate(Jh_d)
  end subroutine find_lowest_energies_only

  subroutine get_device_properties(free_mem_in_bytes)
    use cudafor
    use global, only : ik
    integer(ik) :: mem_total, istat
    integer(ik), intent(out) :: free_mem_in_bytes
    istat = cudaMemGetInfo(free_mem_in_bytes, mem_total)
  end subroutine get_device_properties
end module gpusearch
