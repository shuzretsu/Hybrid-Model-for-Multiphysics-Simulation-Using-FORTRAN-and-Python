# Makefile for compiling Fortran simulations

all: heat_conduction fluid_flow molecular_dynamics

heat_conduction: heat_conduction.f90
    gfortran -o heat_conduction heat_conduction.f90

fluid_flow: fluid_flow.f90
    gfortran -o fluid_flow fluid_flow.f90

molecular_dynamics: molecular_dynamics.f90
    gfortran -o molecular_dynamics molecular_dynamics.f90

clean:
    rm -f *.o heat_conduction fluid_flow molecular_dynamics
