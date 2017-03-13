""" 
This file contains the function that returns the spatial heatflux variation 
The values which are returned correspond to the local value of heatflux at the discretized points.
"""

from calculate_moments.bulk_velocity import calculate_vel_bulk
from calculate_moments.density import calculate_density
import arrayfire as af
import params

N_vel              = params.N_vel

def calculate_heat_flux(f, v):

  deltav            = af.sum(v[0, 1]-v[0, 0])
  vel_bulk          = calculate_vel_bulk(f, v)
  vel_bulk          = af.tile(vel_bulk, 1, N_vel)
  value_of_heatflux = af.sum(f*(v-vel_bulk)**3, 1)*deltav
  value_of_heatflux = value_of_heatflux/calculate_density(f, v)
  
  af.eval(value_of_heatflux)

  return(value_of_heatflux)