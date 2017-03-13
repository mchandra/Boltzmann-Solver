"""
This file holds the function for Dirichlet boundary conditions.
The distribution function array, and velocity arrays are passed to this function.
The function returns the distribution function with the applied B.C.
"""

import params
import numpy as np
import arrayfire as af

""" Here begins the import of parameter variables from params.py """

N_x     = params.N_x
N_ghost = params.N_ghost

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

rho_left  = params.rho_left
rho_right = params.rho_right

T_wall_left  = params.T_wall_left
T_wall_right = params.T_wall_right

bulk_vel_left  = params.bulk_vel_left
bulk_vel_right = params.bulk_vel_right

""" Here ends the import of variables from params.py """

def wall(f, v):

  left_zones  = np.arange(N_ghost)
  right_zones = np.arange((N_x + N_ghost), (N_x + 2 * N_ghost))

  left_zones  = af.to_array(left_zones)
  right_zones = af.to_array(right_zones)

  f[left_zones, :]  = rho_left * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left)) * \
                      af.exp(-mass_particle*(v[left_zones, :] - bulk_vel_left)**2/(2*boltzmann_constant*T_wall_left))

  f[right_zones, :] = rho_right * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right)) * \
                      af.exp(-mass_particle*(v[right_zones, :] - bulk_vel_right)**2/(2*boltzmann_constant*T_wall_right))
  
  af.eval(f)

  return(f)