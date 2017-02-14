# In this file set all the parameters that are to be used:
# This file acts as a master control file, where all the capabilities of the code can be tested:

# Setting velocity and spatial grid points
gridpoints_positions = 20
ghost_zones = 3
gridpoints_velocity  = 100

# Boundaries of domain
left_boundary  = 0
right_boundary = 1.0
length         = right_boundary - left_boundary

# Setting mass of the particle, boltzmann-constant
mass_particle      = 1.0
boltzmann_constant = 1.0

# Scattering time scale
tau   = 1.0
# Magnitude of maximum velocity
# Used for uniform distribution
v_max = 5

# Time Parameters for the simulation:
dt         = 0.05 # Size of the time-step
final_time = 3.0

# Setting up the initial temperature and density in the domain:
initial_temperature = 1.5 # Used to assign MB velocity distribution
initial_density     = 1.0

# In order to set a density perturbation in the domain, one needs to change the function for the initial density 
# which has been defined in initialize.py. The similar function has also been defined for temperature

# Setting the initial distribution function:
# "maxwell-boltzmann"    - Assigns a Maxwell-Boltzmann distribution corresponding to the T_initial
# "uniform-distribution" - Assigns a Normal distribution corresponding to the set v_max
type_distribution_function = "maxwell-boltzmann"

# Wall Options Available:
# "thermal"  - Sets the wall as a thermal boundary with the temperatures specified
# "periodic" - Sets the walls to behave as a periodic B.C
wall_type = "thermal"

if(wall_type == "thermal"):
  T_wall_left  = 2.0
  T_wall_right = 2.0

collisions_enabled = "false" # Set to true or false