module fluid_mod
  implicit none
  contains

  subroutine fluid_flow(U, V, P, n, dt, dx, dy, rho, mu)
    real, dimension(n, n) :: U, V, P
    real :: dt, dx, dy, rho, mu
    integer :: i, j

    ! Update velocity and pressure fields based on Navier-Stokes equations
    do i = 2, n-1
      do j = 2, n-1
        U(i,j) = U(i,j) + dt * (-U(i,j)*(U(i+1,j) - U(i-1,j)) / (2*dx) +
                                  mu * (U(i+1,j) - 2*U(i,j) + U(i-1,j)) / (dx*dx))
        V(i,j) = V(i,j) + dt * (-V(i,j)*(V(i,j+1) - V(i,j-1)) / (2*dy) +
                                  mu * (V(i,j+1) - 2*V(i,j) + V(i,j-1)) / (dy*dy))
        P(i,j) = P(i,j) + dt * (-rho * ((U(i+1,j) - U(i-1,j)) / (2*dx) +
                                        (V(i,j+1) - V(i,j-1)) / (2*dy)))
      end do
    end do
  end subroutine fluid_flow
end module fluid_mod
