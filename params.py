"""
In this file set all the parameters that are to be used:

This file acts as a master control file, where all the capabilities of the code can be tested:
Keep in mind though some parameters may need to be changed from initialize.py. 
It has been explicitly mentioned when such a scenario arises.
"""

# Setting velocity and spatial grid points
N_x     = 501
N_ghost = 3
N_vel   = 101

# Boundaries of domain
left_boundary  = 0
right_boundary = 1.0

# Setting mass of the particle, boltzmann-constant
mass_particle      = 1.0
boltzmann_constant = 1.0

# Scattering time scale
tau     = 0.01
# Magnitude of maximum velocity
vel_max = 1.0

# The vel_max which has been declared above is for declaration of the velocity array.
# It holds the values of velocities of the discretized velocity grid.
# The discretized velocity is used in solving for the distribution function. 

# Time Parameters for the simulation:
dt         = 0.01 # Size of the time-step
final_time = 5.0

# Setting up the initial temperature and density in the domain:
initial_temperature = 1.0 
initial_density     = 1.0

# Note that the above can only hold true when a constant value of density and temperature prevails throughout 
# the entire domain. In case of distributions that vary spatially, changes need to be made to initialize.py

# Setting the initial distribution function:
# "maxwell-boltzmann"    - Assigns a Maxwell-Boltzmann distribution as per declared parameters
# "uniform-distribution" - Assigns a Normal distribution as per declared parameters
type_distribution_function = "maxwell-boltzmann"

# Wall Options Available:
# "dirichlet"     - Sets the wall as a with conditions specified(can be used to set thermal B.C's)
# "periodic"      - Sets the walls to behave as a periodic B.C

wall_type = "periodic"

if(wall_type == "dirichlet"):
  rho_left  = 1.0
  rho_right = 1.0

  T_wall_left  = 1.0
  T_wall_right = 1.0

  bulk_vel_left  = 0
  bulk_vel_right = 0

# Depending on the following condition, the code shall work as a collisional or collisionless code 
collisions_enabled = "false" # Set to true or false