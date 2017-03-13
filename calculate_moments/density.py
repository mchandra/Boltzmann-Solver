""" 
This file contains the function that returns the spatial density variation 
The values which are returned correspond to the local value of density at the discretized points.
"""

def calculate_density(f, v):
  
  deltav           = v[0, 1]-v[0, 0]
  value_of_density = af.sum(f, 1)*deltav
  
  af.eval(value_of_density)
  
  return(value_of_density)