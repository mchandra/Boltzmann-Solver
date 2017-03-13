"""
This file holds the function for periodic boundary conditions.
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

""" Here ends the import of variables from params.py """

def wall(f, v):
  
  left_zones  = np.arange(N_ghost)
  right_zones = np.arange((N_x + N_ghost), (N_x + 2 * N_ghost))  

  left_zones  = af.to_array(left_zones)
  right_zones = af.to_array(right_zones)
  
  f[left_zones, :]  = f[N_x - 2 + N_ghost - left_zones, :]
  f[right_zones, :] = f[right_zones - N_x + 1 , :]

  af.eval(f)

  return(f)