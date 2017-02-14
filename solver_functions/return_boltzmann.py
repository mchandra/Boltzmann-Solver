import numpy as np
import arrayfire as af 
import params
af.set_backend("cpu")

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
from calculate_moments.density import calculate_density
from calculate_moments.temperature import calculate_temperature

def f_MB(x, v, f):
  n = af.tile(af.reorder(calculate_density(f, v)), gridpoints_velocity, 1)
  T = af.tile(af.reorder(calculate_temperature(f, v)), gridpoints_velocity, 1)
  f_MB = n*np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
         af.exp(-mass_particle*v**2/(2*boltzmann_constant*T))
  return(f_MB)