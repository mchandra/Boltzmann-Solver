import numpy as np

def BGK_collision_operator(config, delta_f_hat):
  """
  Returns the array that contains the values of the linearized BGK collision operator.
  The expression that has been used may be understood more clearly by referring to the
  Sage worksheet on https://goo.gl/dXarsP

  Parameters:
  -----------
    config : Class config which is obtained by set() is passed to this file

    delta_f_hat : The array of delta_f_hat which is obtained from each step
                  of the time integration. 

  Output:
  -------
    C_f : Array which contains the values of the linearized collision operator. 

  """

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv    = vel_x[1] - vel_x[0]
  tau   = config.tau

  delta_T_hat   = np.sum(delta_f_hat * (vel_x**2 - temperature_background)) * dv/rho_background
  delta_rho_hat = np.sum(delta_f_hat) * dv
  delta_v_hat   = np.sum(delta_f_hat * vel_x) * dv/rho_background
  
  expr_term_1 = np.sqrt(2 * mass_particle**3) * delta_T_hat * rho_background * vel_x**2
  expr_term_2 = 2 * np.sqrt(2 * mass_particle) * boltzmann_constant * delta_rho_hat * temperature_background**2
  expr_term_3 = 2 * np.sqrt(2 * mass_particle**3) * rho_background * delta_v_hat * vel_x * temperature_background
  expr_term_4 = - np.sqrt(2 * mass_particle) * boltzmann_constant * delta_T_hat * rho_background * temperature_background
  
  C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)*\
         np.exp(-mass_particle * vel_x**2/(2 * boltzmann_constant * temperature_background))/\
         (4 * np.sqrt(np.pi * temperature_background**5 * boltzmann_constant**3)) - delta_f_hat
         )/tau
        )
  
  return C_f
