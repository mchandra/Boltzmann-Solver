""" 
This file contains the function that returns the spatial bulk velocity variation 
The values which are returned correspond to the local value of bulk velocity at the discretized points.
"""

def calculate_vel_bulk(f, v):
  
  deltav            = af.sum(v[0, 1]-v[0, 0])
  value_of_momentum = af.sum(f*v, 1)*deltav
  
  af.eval(value_of_momentum)

  return(value_of_momentum)