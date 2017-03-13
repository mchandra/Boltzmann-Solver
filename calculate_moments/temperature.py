""" 
This file contains the function that returns the spatial temperature variation 
The values which are returned correspond to the local temperature at the discretized points.
"""

from bulk_velocity import calculate_vel_bulk
from density import calculate_density
import params

N_vel              = params.N_vel
mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant


def calculate_temperature(f, v):

  deltav               = af.sum(v[0, 1]-v[0, 0])
  vel_bulk             = calculate_vel_bulk(f, v)
  vel_bulk             = af.tile(vel_bulk, 1, N_vel)
  value_of_temperature = af.sum(f*(v-vel_bulk)**2, 1)*deltav
  value_of_temperature = (mass_particle/boltzmann_constant) *\
                         (value_of_temperature/calculate_density(f, v))

  af.eval(value_of_temperature)
  
  return(value_of_temperature)