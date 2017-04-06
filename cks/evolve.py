import numpy as np
import arrayfire as af
import cks.initialize as initialize

def calculate_density(f, vel_x):
  deltav           = af.sum(vel_x[0, 1]-vel_x[0, 0])
  value_of_density = af.sum(f, 1)*deltav
  
  af.eval(value_of_density)

  return(value_of_density)

def calculate_vbulk(f, vel_x):
  deltav               = af.sum(vel_x[0, 1]-vel_x[0, 0])
  value_of_momentum    = af.sum(f*vel_x, 1)*deltav
  value_of_vbulk       = value_of_momentum/calculate_density(f, vel_x)
  
  af.eval(value_of_vbulk)
  
  return(value_of_vbulk)

def calculate_temperature(f, vel_x):
  deltav               = af.sum(vel_x[0, 1]-vel_x[0, 0])
  v_bulk               = af.tile(calculate_vbulk(f, vel_x), 1, N_velocity)
  value_of_temperature = af.sum(f*(vel_x-v_bulk)**2, 1)*deltav
  value_of_temperature = value_of_temperature/calculate_density(f, vel_x)
  
  af.eval(value_of_temperature)
  
  return(value_of_temperature)

def f_MB(config, f, vel_x):
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  n      = af.tile(calculate_density(f, vel_x), 1, N_velocity)
  T      = af.tile(calculate_temperature(f, vel_x), 1, N_velocity)
  v_bulk = af.tile(calculate_vbulk(f, vel_x), 1, N_velocity)
  
  f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
           af.exp(-mass_particle*(vel_x-v_bulk)**2/(2*boltzmann_constant*T))
      
  af.eval(f_MB)
  
  return(f_MB)

def f_interp(config, dt, x, vel_x, f):
  x_new     = x - vel_x*dt
  step_size = af.sum(x[1,0] - x[0,0])
  
  f_interp  = af.constant(0, N_positions + 2*ghost_zones, N_velocity)
  f_interp  = af.Array.as_type(f_interp, af.Dtype.f64)
  
  # Interpolating:
  
  x_temp = x_new[ghost_zones:-ghost_zones, :].copy()
  
  while(af.sum(x_temp<left_boundary)!=0):
      x_temp = af.select(x_temp<left_boundary,
                         x_temp + length,
                         x_temp
                        )
  while(af.sum(x_temp>right_boundary)!=0):
      x_temp = af.select(x_temp>right_boundary,
                         x_temp - length,
                         x_temp
                        )

  x_temp        = af.Array.as_type(x_temp, af.Dtype.f64)
  
  x_interpolant = x_temp/step_size + ghost_zones
  
  x_interpolant = af.Array.as_type(x_interpolant, af.Dtype.f64)
  f             = af.Array.as_type(f, af.Dtype.f64)
  
  f_interp[ghost_zones:-ghost_zones, :] = af.approx1(f, x_interpolant,\
                                                     af.INTERP.CUBIC_SPLINE
                                                    )
  
  f_interp          = af.Array.as_type(f_interp, af.Dtype.f64)
  
  af.eval(f_interp)
  return f_interp

for time_index, t0 in enumerate(time_array):
    # if(time_index%10==0):
    #     print("Physical Time            = ", t0)
    # We shall split the Boltzmann-Equation and solve it:
    # In this step we are solving the collisionless equation
    fstar = f_interp(dt, x, v, f_current)

    fstar[:ghost_zones,:]                = fstar[-(2*ghost_zones + 1):-(ghost_zones + 1)]
    fstar[N_positions + ghost_zones:, :] = fstar[(ghost_zones + 1):(2*ghost_zones + 1)]
        
    f_current = fstar
    
    data[time_index] = np.max(calculate_density(f_current, v))