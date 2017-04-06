import numpy as np
import arrayfire as af 

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

  config.N_vel_x        = params.size['N_vel_x']
  config.N_x            = params.size['N_x']
  config.vel_x_max      = params.size['vel_x_max']
  config.N_ghost_x      = params.size['N_ghost_x']
  config.left_boundary  = params.size['left_boundary']
  config.right_boundary = params.size['right_boundary']

  config.final_time = params.time['final_time']
  config.dt         = params.time['dt']
    
  config.fields_enabled  = params.EM_fields['enabled']
  config.charge_particle = params.EM_fields['charge_particle']

  config.collisions_enabled = params.collisions['enabled']
  config.collision_operator = params.collisions['collision_operator']
  config.tau                = params.collisions['tau']

  return config

def calculate_x(config):
  """
  Returns the 2D array of x which is used in the computations of the Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    x : Array holding the values of x tiled along axis 1
  """
  N_x       = config.N_x
  N_vel_x   = config.N_vel_x
  N_ghost_x = config.N_ghost_x

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary

  x  = np.linspace(left_boundary, right_boundary, N_x)
  dx = x[1] - x[0]

  x_ghost_left  = np.linspace(-(N_ghost_x)*dx + left_boundary, left_boundary - dx, N_ghost_x)
  x_ghost_right = np.linspace(right_boundary + dx, right_boundary + N_ghost_x*dx , N_ghost_x)

  x  = np.concatenate([x_ghost_left, x, x_ghost_right])
  x  = af.Array.as_type(af.to_array(x), af.Dtype.f64)
  x  = af.tile(x, 1, N_vel_x)

  af.eval(x)
  return x

def calculate_vel_x(config):
  """
  Returns the 2D array of vel_x which is used in the computations of the Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    x : Array holding the values of vel_x tiled along axis 0
  """
  N_x       = config.N_x
  N_vel_x   = config.N_vel_x
  N_ghost_x = config.N_ghost_x
  vel_x_max = config.vel_x_max

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)
  vel_x = af.tile(af.reorder(vel_x), N_x + 2*N_ghost_x, 1)

  af.eval(vel_x)
  return vel_x


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
                   of vel_x and x
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_x                  = calculate_vel_x(config)

  f_background = rho_background * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                 af.exp(-mass_particle*vel_x**2/(2*boltzmann_constant*temperature_background))

  af.eval(f_background)
  return f_background

def f_initial(config):
  """
  Returns the value of f_initial, depending on the parameters set in 
  the config object

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    f_initial : Array which contains the values of f_initial at different values
                of vel_x and x
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_x                  = calculate_vel_x(config)

  pert_x_real = config.pert_x_real
  pert_x_imag = config.pert_x_imag
  k_x         = config.k_x

  x   = calculate_x(config)
  rho = rho_background + (pert_x_real * af.cos(2*np.pi*x) - pert_x_imag * af.sin(2*np.pi*x))

  f_initial = rho * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
              af.exp(-mass_particle*vel_x**2/(2*boltzmann_constant*temperature_background))

  af.eval(f_initial)
  return f_initial

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