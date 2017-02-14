import params
import numpy as np
import arrayfire as af 
import h5py
af.set_backend("cpu")

# In this file set all the parameters that are to be used:
# This file acts as a master control file, where all the capabilities of the code can be tested:

""" Here begins the import of parameter variables from params.py """
gridpoints_positions = params.gridpoints_positions
ghost_zones = params.ghost_zones
gridpoints_velocity  = params.gridpoints_velocity

left_boundary  = params.left_boundary
right_boundary = params.right_boundary
length         = right_boundary - left_boundary

mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant

tau   = params.tau
v_max = params.v_max

dt         = params.dt
final_time = params.final_time

type_distribution_function = params.type_distribution_function

wall_type = params.wall_type

if(wall_type == "thermal"):
  T_wall_left  = params.T_wall_left
  T_wall_right = params.T_wall_right

collisions_enabled = params.collisions_enabled

""" Here ends the import of variables from params.py """

""" Reading the file for initial conditions """

h5f       = h5py.File('data_files/initial_conditions/initial_data.h5', 'r')
time      = h5f['time'][:]
x         = h5f['x'][:]
v         = h5f['vel'][:]
f_initial = h5f['f_initial'][:]
h5f.close()

""" Converting the files to arrayfire's native format : """

x         = af.to_array(x)
v         = af.to_array(v)
f_initial = af.to_array(f_initial)

""" Solving: """

# Setting the initial value to f_current:
f_current = f_initial

for time_index, t0 in enumerate(time):
  
  print("Computing For Time Index = ", time_index)
  print("Physical Time            = ", t0)

  # We shall split the Boltzmann-Equation and solve it:

  # In this step we are solving the collisionless equation:
  from solver_functions.interpolate import f_interp
  fstar = f_interp(dt, x, v, f_current)

  if(collisions_enabled == "true"):
    # We turn off the term v(df/dx) for the following two steps
    from solver_functions.return_boltzmann import f_MB
    f0             = f_MB(x, v, fstar)
    f_intermediate = fstar - (dt/2)*(fstar          - f0)/tau
    f_new          = fstar - (dt)  *(f_intermediate - f0)/tau
    f_new          = fstar

  if(collisions_enabled == "false"):
    f_new = fstar
  
  from boundary_conditions.thermal_boundary import thermal_BC
  f_current = thermal_BC(f_new, v)

  from calculate_moments.temperature import calculate_temperature
  T = af.sum(f_current * v**2, 0)
  print(T)

  h5f = h5py.File('data_files/timestepped_data/solution_'+str(time_index)+'.h5', 'w')
  h5f.create_dataset('f',    data = f_current)
  h5f.create_dataset('x',    data = x)
  h5f.create_dataset('v',    data = v)
  h5f.create_dataset('time', data = time)
  h5f.close()