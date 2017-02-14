import params
import numpy as np
import arrayfire as af 
import h5py
af.set_backend("cpu")

# In this file set all the parameters that are to be used:
# This file acts as a master control file, where all the capabilities of the code can be tested:

""" Here begins the import of parameter variables from params.py """
gridpoints_positions = params.gridpoints_positions
ghost_zones = params.ghost_zones
gridpoints_velocity  = params.gridpoints_velocity

left_boundary  = params.left_boundary
right_boundary = params.right_boundary
length         = right_boundary - left_boundary

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

tau   = params.tau
v_max = params.v_max

dt         = params.dt
final_time = params.final_time

type_distribution_function = params.type_distribution_function

wall_type = params.wall_type

if(wall_type == "thermal"):
  T_wall_left  = params.T_wall_left
  T_wall_right = params.T_wall_right

collisions_enabled = params.collisions_enabled

""" Here ends the import of variables from params.py """

# Definition of the time-array:
time = np.arange(dt, final_time, dt)

# Setting up of spatial and velocity grids:
x  = np.linspace(left_boundary, right_boundary, gridpoints_positions)
dx = x[1] - x[0]

# Obtaining the coordinates for the ghost-zones:
x_ghost_left  = np.linspace(-(ghost_zones)*dx + left_boundary, left_boundary - dx, ghost_zones)
x_ghost_right = np.linspace(right_boundary + dx, right_boundary + ghost_zones*dx , ghost_zones)

# Combining them to obtain the entire spatial grid
x  = np.concatenate([x_ghost_left, x, x_ghost_right])

# Obtaining the velocity grid
v  = np.linspace(-v_max, v_max, gridpoints_velocity)

# Converting the arrays to arrayfire's native array format:
x = af.to_array(x)
v = af.to_array(v)

# Tiling to allow for easy vectorization
x = af.tile(af.reorder(x), gridpoints_velocity, 1)
v = af.tile(v, 1, gridpoints_positions + 2*ghost_zones)

# Definitions for initial temperature and initial density arrays:
# Changes need to be here for perturbations in temperature and density:

initial_temperature = af.data.constant(params.initial_temperature, gridpoints_velocity, gridpoints_positions + 2*ghost_zones)
initial_density     = af.data.constant(params.initial_density, gridpoints_velocity, gridpoints_positions + 2*ghost_zones)


# Intializing the values for distribution function:
if(type_distribution_function == "maxwell-boltzmann"):
  f_initial = initial_density * af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*initial_temperature)) * \
                                af.exp(-mass_particle*v**2/(2*boltzmann_constant*initial_temperature))

if(type_distribution_function == "uniform-distribution"):
  f_initial = (initial_density/(2*v_max)) * af.data.constant(1, gridpoints_velocity, gridpoints_positions + 2*ghost_zones)

# Setting the distribution at the walls:
left_zones  = np.arange(ghost_zones)
right_zones = np.arange((gridpoints_positions + ghost_zones), (gridpoints_positions + 2 * ghost_zones))

for i in left_zones:
  f_initial[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left)) * \
                    af.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_wall_left))

for i in right_zones:
  f_initial[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right)) * \
                    af.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_wall_right))

# Writing all the data to a file:

h5f = h5py.File('data_files/initial_conditions/initial_data.h5', 'w')
h5f.create_dataset('time',      data = time)
h5f.create_dataset('x',         data = x)
h5f.create_dataset('vel',       data = v)
h5f.create_dataset('f_initial', data = f_initial)
h5f.close()