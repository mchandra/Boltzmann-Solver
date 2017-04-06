import numpy as np

constants = dict(
                  mass_particle      = 1.0,
                  boltzmann_constant = 1.0,
                )

background_electrons = dict(
                            rho         = 1.0, 
                            temperature = 1.0, 
                            vel_bulk    = 0
                           )

background_ions = dict(
                       rho         = 1.0, 
                       temperature = 1.0, 
                       vel_bulk    = 0
                      )

perturbation = dict(
                    pert_x_real = 1e-2, 
                    pert_x_imag = 0,
                    pert_y_real = 0,
                    pert_y_imag = 0,
                    k_x         = 2*np.pi,
                    k_y         = 2*np.pi 
                   ) 

size = dict(
            N_vel_x   = 1001,
            N_x       = 32,
            vel_x_max = 5.0,
            N_vel_y   = 1001,
            N_y       = 32,
            vel_y_max = 5.0
           )

time = dict(
            final_time   = 5.0,
            dt           = 0.01
           )

EM_fields = dict(
                 enabled         = 'False',
                 charge_particle = -10.0
                )

collisions = dict(
                  enabled            = 'False',
                  collision_operator = 'BGK',
                  tau                =  0.01
                 )