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

def thermal_BC(f, v):
  left_zones  = np.arange(ghost_zones)
  right_zones = np.arange((gridpoints_positions + ghost_zones), (gridpoints_positions + 2 * ghost_zones))

  for i in left_zones:
    f[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left)) * \
              af.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_wall_left))

  for i in right_zones:
    f[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right)) * \
              af.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_wall_right))

  return(f)