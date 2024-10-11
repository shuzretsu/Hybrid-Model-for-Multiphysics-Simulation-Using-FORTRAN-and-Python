module heat_mod
  implicit none
  contains

  subroutine heat_conduction(T, n, dt, dx, kappa)
    real, dimension(n, n) :: T
    real :: dt, dx, kappa
    integer :: i, j

    do i = 2, n-1
      do j = 2, n-1
        T(i,j) = T(i,j) + kappa * dt * ((T(i+1,j) - 2*T(i,j) + T(i-1,j)) / (dx*dx) + 
                                        (T(i,j+1) - 2*T(i,j) + T(i,j-1)) / (dx*dx))
      end do
    end do
  end subroutine heat_conduction
end module heat_mod