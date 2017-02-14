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

from boundary_conditions.thermal_boundary import thermal_BC

def f_interp(dt, x, v, f):
  x_new      = x - v*dt
  step_size  = af.sum(x[1] - x[0]) # To convert to scalar
  f_interp   = af.approx1(af.reorder(f), af.reorder(x_new/step_size), af.INTERP.CUBIC)
  f_interp   = af.reorder(f_interp)

  left_zones     = af.where(x_new<left_boundary)
  left_zones_row = left_zones%(gridpoints_velocity)
  left_zones_col = left_zones/(gridpoints_velocity) 

  f_interp[left_zones_row, left_zones_col] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left)) * \
                                             af.exp(-mass_particle*v[left_zones_row, left_zones_col]**2/(2*boltzmann_constant*T_wall_left))

  right_zones     = af.where(x_new>right_boundary)
  right_zones_row = right_zones%(gridpoints_velocity)
  right_zones_col = right_zones/(gridpoints_velocity)

  f_interp[right_zones_row, right_zones_col] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right)) * \
                                               af.exp(-mass_particle*v[right_zones_row, right_zones_col]**2/(2*boltzmann_constant*T_wall_right))

  return(f_interp)