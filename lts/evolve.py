import numpy as np
from scipy.integrate import odeint
import lts.initialize as initialize
from lts.collision_operators import BGK_collision_operator

def ddelta_f_hat_dt(Y, t, config):
  """
  Returns the value of the derivative of the mode expansion of the perturbation in 
  distribution function with respect to time. This is used as the input function for
  odeint which is utilized in the time integration function to evolve the system.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

    Y: An array passed to the function which has the first N_vel elements consisting
       of the real parts of the mode expansion of the distribution function with the 
       remaining elements consisting of the imaginary function. This is done owing to the 
       fact that odeint doesn't allow imaginary values.

    t: Time interval over which the derivative is computed for. The value returned by this
       funtion is then integrated over this time interval.

  Output:
  -------
    dYdt : Array which contains the values of the derivative of the Fourier mode expansion of 
           the perturbation in the distribution function with respect to time.

  """

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  
  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x  = vel_x[1] - vel_x[0]

  k_x = config.k_x   
  
  fields_enabled  = config.fields_enabled
  charge_particle = config.charge_particle

  collisions_enabled = config.collisions_enabled 
  tau                = config.tau

  delta_f_hat_real = Y[:vel_x.size]
  delta_f_hat_imag = Y[vel_x.size:]

  delta_f_hat   = delta_f_hat_real + 1j*delta_f_hat_imag

  delta_rho_hat = np.sum(delta_f_hat) * dv_x

  if(fields_enabled!="True"):

    fields_term = 0

  else:

    dfdv_background = initialize.dfdv_background(config)
    delta_E_hat     = -charge_particle * (delta_rho_hat)/(1j * k_x)
    fields_term     = (charge_particle / mass_particle) * delta_E_hat * dfdv_background

  if(collisions_enabled!="True"):

    C_f = 0

  else:

    C_f   = BGK_collision_operator(config, delta_f_hat)

  dYdt = np.concatenate([(k_x * vel_x * delta_f_hat.imag)  -
                         fields_term.real + C_f.real,\
                         -(k_x * vel_x * delta_f_hat.real) +
                         fields_term.imag + C_f.imag \
                       ], axis = 0)
  
  return dYdt

def time_integration(config, delta_f_hat_initial, time_array):
  """
  Performs the time integration for the simulation. This is the main function that
  evolves the system in time. The parameters this function evolves for are dictated
  by the parameters as has been set in the config object. Final distribution function
  and the array that shows the evolution of rho_hat is returned by this function.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

    delta_f_hat_initial : Array containing the initial values of the delta_f_hat. The value
                          for this function is typically obtained from the appropriately named 
                          function from the initialize submodule.

    time_array : Array with consists of all the points at which we are evolving the system for.
                 Data such as the mode amplitude of the density perturbation is also computed at 
                 the time points.

  Output:
  -------
    density_data : The value of the amplitude of the mode expansion of the density perturbation computed at
                   the various points in time as declared in time_array

    new_delta_f_hat : This value that is returned by the function is the distribution function that is obtained at
                      the final time-step. This is particularly useful in cases where comparisons need to be made 
                      between results of the Cheng-Knorr and the linear theory codes.
  
  """
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  N_x       = config.N_x

  x       = np.linspace(0, 1, N_x)
  k_x     = config.k_x 
  vel_x   = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x    = vel_x[1] - vel_x[0]  

  density_data = np.zeros(time_array.size)

  for time_index, t0 in enumerate(time_array):
    t0 = time_array[time_index]
    if (time_index == time_array.size - 1):
        break
    t1 = time_array[time_index + 1]
    t = [t0, t1]

    if(time_index != 0):
      delta_f_hat_initial = old_delta_f_hat.copy()
        
    Y = np.zeros(2*N_vel_x)

    Y[:N_vel_x] = delta_f_hat_initial.real
    Y[N_vel_x:] = delta_f_hat_initial.imag

    Y_new = odeint(ddelta_f_hat_dt, Y, t, args = (config,),\
                   rtol = 1e-20, atol = 1e-15
                  )[1]
    
    new_delta_f_hat = Y_new[:N_vel_x] + 1j*Y_new[N_vel_x:]
    delta_rho_hat   = np.sum(new_delta_f_hat)*dv_x
    
    density_data[time_index] = np.max(delta_rho_hat.real * np.cos(k_x*x) - delta_rho_hat.imag * np.sin(k_x*x))
    old_delta_f_hat          = new_delta_f_hat.copy()

  return(density_data, new_delta_f_hat)
