"""
This is the code which upon running will write a file, that contains the values for the initial conditions.
This file uses the data as declared in params.py to initialize the data.
In case of spatially varying functions however, changes need to be directly made in this file.
"""

import params
import numpy as np
import arrayfire as af 
import h5py
af.set_backend("cpu")

""" Here begins the import of parameter variables from params.py """
N_x     = params.N_x
N_ghost = params.N_ghost
N_vel   = params.N_vel

left_boundary  = params.left_boundary
right_boundary = params.right_boundary

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

tau     = params.tau 
vel_max = params.vel_max

dt         = params.dt
final_time = params.final_time

initial_temperature = params.initial_temperature
initial_density     = params.initial_density

type_distribution_function = params.type_distribution_function
wall_type                  = params.wall_type

if(wall_type == "dirichlet"):
  rho_left  = params.rho_left
  rho_right = params.rho_right

  T_wall_left  = params.T_wall_left
  T_wall_right = params.T_wall_right

  bulk_vel_left  = params.bulk_vel_left
  bulk_vel_right = params.bulk_vel_right

collisions_enabled = params.collisions_enabled

""" Here ends the import of variables from params.py """

# Definition of the time-array:
time = np.arange(dt, final_time, dt)

# Setting up of spatial and velocity grids:
x  = np.linspace(left_boundary, right_boundary, N_x)
dx = x[1] - x[0]

# Obtaining the coordinates for the ghost-zones:
x_ghost_left  = np.linspace(-(N_ghost)*dx + left_boundary, left_boundary - dx, N_ghost)
x_ghost_right = np.linspace(right_boundary + dx, right_boundary + N_ghost*dx , N_ghost)

# Combining them to obtain the entire spatial grid
x  = np.concatenate([x_ghost_left, x, x_ghost_right])

# Obtaining the velocity grid
v  = np.linspace(-vel_max, vel_max, N_vel)

# Converting the arrays to arrayfire's native array format:
x = af.to_array(x)
v = af.to_array(v)

# Tiling to allow for easy vectorization
x = af.tile(x, 1, N_vel)
v = af.tile(af.reorder(v), N_x + 2*N_ghost, 1)

# Declaring an array which will hold the values of bulk velocities:
# The following conditions assign half the domain with the left bulk velocity.
# With the remaining half taking the values of the right side bulk velocity.
# These conditions are particularly useful for the stationary-shock test.

v_bulk = af.constant(0, N_x + 2*N_ghost, N_vel)

if(wall_type == "dirichlet"):
  v_bulk[:N_ghost + 0.5*N_x, :]  = bulk_vel_left
  v_bulk[N_ghost +  0.5*N_x:, :] = bulk_vel_right

# Definitions for initial temperature and initial density arrays:
# Changes need to be HERE for spatial variations in temperature and density:

initial_temperature = af.data.constant(initial_temperature, N_x + 2*N_ghost, N_vel)
initial_density     = af.data.constant(initial_density, N_x + 2*N_ghost, N_vel)

initial_density = 1 + 0.5 * af.sin(2*np.pi*x)

# Intializing the values for distribution function:
if(type_distribution_function == "maxwell-boltzmann"):
  f_initial = initial_density * af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*initial_temperature)) * \
                                af.exp(-mass_particle*(v - v_bulk)**2/(2*boltzmann_constant*initial_temperature))

if(type_distribution_function == "uniform-distribution"):
  # max_vel below is used for finding the maximum velocity for a uniform distribution.
  # This is calculated on the basis of the value of density and temperature.
  # Note: This is unrelated to the vel_max that has been defined for the velocity array
  max_vel   = np.sqrt(3*initial_temperature*boltzmann_constant/mass_particle)
  f_initial = (initial_density/(2*max_vel)) * af.data.constant(1, N_x + 2*N_ghost, N_vel)

# Setting the distribution at the walls, as per the declared B.C:
left_zones  = np.arange(N_ghost)
right_zones = np.arange((N_x + N_ghost), (N_x + 2 * N_ghost))

if(wall_type == "dirichlet"):
  from boundary_conditions.dirichlet_boundary import wall
  f_initial = wall(f_initial, v)

if(wall_type == "periodic"):
  from boundary_conditions.periodic_boundary import wall
  f_initial = wall(f_initial, v)

# The following initial conditions are for the shock test:
# Comment these out/in depending on the need:

# f_initial[:N_ghost + 0.5*N_x, :] = (rho_left  * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left)) * \
#                                     af.exp(-mass_particle*(v - v_bulk)**2/(2*boltzmann_constant*T_wall_left))
#                                    )[:N_ghost + 0.5*N_x, :]

# f_initial[N_ghost + 0.5*N_x:, :] = (rho_right * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right)) * \
#                                     af.exp(-mass_particle*(v - v_bulk)**2/(2*boltzmann_constant*T_wall_left))
#                                    )[N_ghost + 0.5*N_x:, :] 

# Writing all the data to a file:
h5f = h5py.File('data_files/initial_conditions/initial_data.h5', 'w')
h5f.create_dataset('time',      data = time)
h5f.create_dataset('x',         data = x)
h5f.create_dataset('vel',       data = v)
h5f.create_dataset('f_initial', data = f_initial)
h5f.close()