""" 
This is the file that needs to be run in order to get the perform the simulation calculations.
Parameters used in the simulation are those set in params.py
This file generates output files under the data_files/ which is acted upon by the post-processor.
Meaningful plots and movies can be made by making use of the post-processing scripts.
"""
import params
import numpy as np
import arrayfire as af 
import h5py
af.set_backend("cpu")

""" Here begins the import of parameter variables from params.py """
N_x     = params.N_x
N_ghost = params.N_ghost
N_vel   = params.N_vel

left_boundary  = params.left_boundary
right_boundary = params.right_boundary

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

tau     = params.tau 
vel_max = params.vel_max

dt         = params.dt
final_time = params.final_time

initial_temperature = params.initial_temperature
initial_density     = params.initial_density

type_distribution_function = params.type_distribution_function
wall_type                  = params.wall_type

if(wall_type == "dirichlet"):
  rho_left  = params.rho_left
  rho_right = params.rho_right

  T_wall_left  = params.T_wall_left
  T_wall_right = params.T_wall_right

  bulk_vel_left  = params.bulk_vel_left
  bulk_vel_right = params.bulk_vel_right

collisions_enabled = params.collisions_enabled
""" Here ends the import of variables from params.py """

""" Reading the file for initial conditions """

h5f       = h5py.File('data_files/initial_conditions/initial_data.h5', 'r')
time      = h5f['time'][:]
x         = h5f['x'][:]
v         = h5f['vel'][:]
f_initial = h5f['f_initial'][:]
h5f.close()

""" Declaring the arrays which hold time progressed data of moments of the distribution function"""

density_data     = np.zeros(time.size)
v_bulk_data      = np.zeros(time.size) 
temperature_data = np.zeros(time.size)
heatflux_data    = np.zeros(time.size)

""" Converting the files to arrayfire's native format : """

x         = af.to_array(x)
v         = af.to_array(v)
f_initial = af.to_array(f_initial)

""" Importing of the functions which shall be used in solving"""

# This function is used to perform the interpolation 
from solver_functions.interpolate import f_interp

# The following function needs to be imported in the collisional case:
if(collisions_enabled == "true"):
  from solver_functions.return_boltzmann import f_MB

if(wall_type == "periodic"):
  from boundary_conditions.periodic_boundary import wall

if(wall_type == "dirichlet"):
  from boundary_conditions.dirichlet_boundary import wall

# For saving averaged bulk parameters:
from calculate_moments.density import calculate_density
from calculate_moments.bulk_velocity import calculate_vel_bulk
from calculate_moments.temperature import calculate_temperature
from calculate_moments.heat_flux import calculate_heat_flux

""" Solving: """

# Setting the initial value to f_current to proceed with the simulation:

f_current = f_initial

for time_index, t0 in enumerate(time):
  
  print("Computing For Time Index = ", time_index)
  print("Physical Time            = ", t0)
  print() # Leaving an empty line for readability.

  # Splitting the Boltzmann-Equation and solving it:

  # In this step we are solving the collisionless equation:
  fstar = f_interp(dt, x, v, f_current)  
  fstar = wall(fstar, v) # Applying B.C's

  if(collisions_enabled == "true"):
    
    # We turn off the term v(df/dx) for the following two steps
    f0             = f_MB(x, v, fstar)
    f_intermediate = fstar - (dt/2)*(fstar          - f0)/tau
    f_new          = fstar - (dt)  *(f_intermediate - f0)/tau
    f_new          = fstar

  if(collisions_enabled == "false"):
    f_new = fstar
    
  f_current = wall(f_new, v)
  
  density_data[time_index]     = af.sum(calculate_density(f_current, v)[N_ghost:-N_ghost])/N_x
  v_bulk_data[time_index]      = af.sum(calculate_vel_bulk(f_current, v)[N_ghost:-N_ghost])/N_x
  temperature_data[time_index] = af.sum(calculate_temperature(f_current, v)[N_ghost:-N_ghost])/N_x
  heatflux_data[time_index]    = af.sum(calculate_heat_flux(f_current, v)[N_ghost:-N_ghost])/N_x

  if(time_index%1000 == 0):
    h5f = h5py.File('data_files/timestepped_data/solution_'+str(time_index)+'.h5', 'w')

    h5f.create_dataset('f'          , data = f_current)
    h5f.create_dataset('position'   , data = x)
    h5f.create_dataset('velocity'   , data = v)
    h5f.create_dataset('time'       , data = time)

    # Storing all the variation in the bulk parameters:

    h5f.create_dataset('density',       data = density_data)
    h5f.create_dataset('bulk_velocity', data = v_bulk_data)
    h5f.create_dataset('temperature',   data = temperature_data)
    h5f.create_dataset('heat_flux',     data = heatflux_data)

    h5f.close()

h5f = h5py.File('data_files/final_timestep_data/final_timestep.h5', 'w')

h5f.create_dataset('f'          , data = f_current)
h5f.create_dataset('position'   , data = x)
h5f.create_dataset('velocity'   , data = v)
h5f.create_dataset('time'       , data = time)

# Storing all the variation in the bulk parameters:

h5f.create_dataset('density',       data = density_data)
h5f.create_dataset('bulk_velocity', data = v_bulk_data)
h5f.create_dataset('temperature',   data = temperature_data)
h5f.create_dataset('heat_flux',     data = heatflux_data)

h5f.close()
