# Excited_state_solver
This program is an improved version of the earlier GitHub program 'meanstate', and can also calculate approximations for the wave functions and energy eigenvalues of 2D quantum systems.

The application can find an approximation for an excited state wave function without sequential computation of the lower energy states from ground state up.
The source code files are compiled in Linux terminal as in (for the 2d rectangle example)

g++ -O3 -ffast-math rectangle.cpp

The plot.py script is for plotting the resulting output files from the rectangular potential well example, and the grid size parameters at lines 75-77 have to be edited for plotting the 
supercircle results (there the domain side length is L=3.4 and the number of grid points at the end of the simulation is 953x953).
