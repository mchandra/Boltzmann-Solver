import numpy as np 

class config:
  pass

def set(params):
  """
  Used to set the parameters that are used in the simulation

  Parameters:
  -----------
    params : Name of the file that contains the parameters for the simulation run
             is passed to this function. 

  Output:
  -------
    config : Object whose attributes contain all the simulation parameters. This is
             passed to the remaining solver functions.
  """
  
  config.mass_particle      = params.constants['mass_particle']
  config.boltzmann_constant = params.constants['boltzmann_constant']

  config.rho_background         = params.background_electrons['rho']
  config.temperature_background = params.background_electrons['temperature']
  config.vel_bulk_background    = params.background_electrons['vel_bulk']

  config.pert_x_real = params.perturbation['pert_x_real']
  config.pert_x_imag = params.perturbation['pert_x_imag']
  config.k_x         = params.perturbation['k_x']

  config.N_vel_x   = params.size['N_vel_x']
  config.N_x       = params.size['N_x']
  config.vel_x_max = params.size['vel_x_max']

  config.final_time = params.time['final_time']
  config.dt         = params.time['dt']
    
  config.fields_enabled  = params.EM_fields['enabled']
  config.charge_particle = params.EM_fields['charge_particle']

  config.collisions_enabled = params.collisions['enabled']
  config.collision_operator = params.collisions['collision_operator']
  config.tau                = params.collisions['tau']

  return config

def f_background(config):
  """
  Returns the value of f_background, depending on the parameters set in 
  the config class

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

  Output:
  -------
    f_background : Array which contains the values of f_background at different values
                   of vel_x.
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)

  f_background = rho_background * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                 np.exp(-mass_particle*vel_x**2/(2*boltzmann_constant*temperature_background))

  return f_background

def dfdv_background(config):
  """
  Returns the value of the derivative of f_background w.r.t to vel_x, depending 
  on the parameters set in the config class. NOTE : Currently only valid when
  f_background is Maxwellian

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

  Output:
  -------
    dfdv_background : Array which contains the values of dfdv_background at different values
                      of vel_x.
  """

  # NOTE: Numerical Differentiation to be added to allow arbitrary choice
  # of background distribution function

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)

  dfdv_background = f_background(config) *\
                    (-mass_particle*vel_x)/(boltzmann_constant*temperature_background)

  return dfdv_background

def time_array(config):
  """
  Returns the value of the time_array at which we solve for in the simulation. 
  The time_array is set depending on the options which have been mention in config.

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

  Output:
  -------
    time_array : Array that contains the values of time at which the 
                 simulation evaluates the physical quantities. 

  """
  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(0, final_time + dt, dt)

  return time_array

def init_delta_f_hat(config):
  """
  Returns the initial value of delta_f_hat which is setup depending on
  the perturbation parameters set in config. 

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

  Output:
  -------
    delta_f_hat_initial : Array which contains the values of initial mode perturbation 
                          in the distribution function.

  """

  pert_x_real = config.pert_x_real 
  pert_x_imag = config.pert_x_imag 

  N_vel_x   = config.N_vel_x
  vel_x_max = config.vel_x_max
  
  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)

  delta_f_hat_initial = np.zeros(N_vel_x)

  delta_f_hat_initial = pert_x_real*f_background(config) +\
                        pert_x_imag*f_background(config)*1j 

  return delta_f_hat_initial