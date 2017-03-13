"""
This file contains the interpolation function which shall be used to solve for the collisionless part of the equation. 
This function performs the interpolation depending upon the set conditions for the walls.
"""

import params
import numpy as np
import arrayfire as af 

""" Here begins the import of parameter variables from params.py """
N_x     = params.N_x
N_ghost = params.N_ghost
N_vel   = params.N_vel

left_boundary  = params.left_boundary
right_boundary = params.right_boundary
length         = right_boundary - left_boundary

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

dt        = params.dt

wall_type = params.wall_type

if(wall_type == "dirichlet"):
  rho_left  = params.rho_left
  rho_right = params.rho_right

  T_wall_left  = params.T_wall_left
  T_wall_right = params.T_wall_right

  bulk_vel_left  = params.bulk_vel_left
  bulk_vel_right = params.bulk_vel_right

""" Here ends the import of variables from params.py """

def f_interp(dt, x, v, f):
  x_new     = x - (v * dt)
  step_size = af.sum(x[1,0] - x[0,0])
  f_inter   = af.constant(0, (N_x + 2*N_ghost), N_vel)

  # Interpolating:
  
  if(wall_type == "periodic"):
    x_temp = af.select(x_temp<left_boundary,
                       x_temp + length,
                       x_temp
                      )
        
    x_temp = af.select(x_temp>right_boundary,
                       x_temp - length,
                       x_temp
                      )

  x_temp             = x_new[N_ghost:-N_ghost, :].copy()
  interpolant_values = x_new[N_ghost:-N_ghost, :]/step_size + N_ghost
  
  f_inter[N_ghost:-N_ghost, :] = af.approx1(f, interpolant_values, af.INTERP.CUBIC)
  
  if(wall_type == "dirichlet"):
    f_left  = rho_left * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_left))*\
              af.exp(-mass_particle*(v - bulk_vel_left)**2/(2*boltzmann_constant*T_wall_left))
    
    f_right = rho_right * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_wall_right))*\
              af.exp(-mass_particle*(v - bulk_vel_right)**2/(2*boltzmann_constant*T_wall_right))
    
    f_inter[N_ghost:-N_ghost,:] = af.select(x_new[N_ghost:-N_ghost, :]<=left_boundary, \
                                            f_left[N_ghost:-N_ghost,:], \
                                            f_inter[N_ghost:-N_ghost,:] \
                                           )
    
    f_inter[N_ghost:-N_ghost,:] = af.select(x_new[N_ghost:-N_ghost, :]>=right_boundary, \
                                            f_right[N_ghost:-N_ghost,:], \
                                            f_inter[N_ghost:-N_ghost,:]
                                           )
    


  af.eval(f_inter)

  return f_inter
