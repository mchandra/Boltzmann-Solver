"""
This file contains the definition of the function that returns the local maxwell boltzmann distribution.
The parameters that are passed to this function are the position, velocity and distribution function arrays.
The array returned is the Maxwell-Boltzmann distribution having the same bulk parameters as the passed f array.
"""
import params
import numpy as np
import arrayfire as af 

""" Here begins the import of parameter variables from params.py """
N_x     = params.N_x
N_ghost = params.N_ghost
N_vel   = params.N_vel

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

""" Here ends the import of variables from params.py """

from calculate_moments.density import calculate_density
from calculate_moments.temperature import calculate_temperature
from calculate_moments.bulk_velocity import calculate_vel_bulk

def f_MB(x, v, f):
 
  n      = af.tile(calculate_density(f, v), 1, N_velocity)
  T      = af.tile(calculate_temperature(f, v), 1, N_velocity)
  v_bulk = af.tile(calculate_vel_bulk(f, v), 1, N_velocity)
  
  f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
           af.exp(-mass_particle*(v-v_bulk)**2/(2*boltzmann_constant*T))

  af.eval(f_MB)
 
  return(f_MB)
