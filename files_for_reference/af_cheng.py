import numpy as np
import arrayfire as af
af.set_backend("cpu")

N_positions = 100
N_velocity  = 2000
length      = 1.0

mass_particle      = 1.0
boltzmann_constant = 1.0

# Scattering time scale
tau = 1.0

# Time Parameters for the simulation:
dt         = 0.001 # Size of the time-step
final_time = 5.0
time       = np.arange(dt, final_time, dt)

# Setting up the temperature parameters for the simulations:
T_initial = 1.5
T_walls   = 2.0

# Setting up of spatial and velocity grids:
x = np.linspace(0, 1, N_positions)
v = np.linspace(-100, 100, N_velocity)

# Converting to arrayfire array's:
x = af.to_array(x)
v = af.to_array(v)

# Conversion to allow for easy vectorization
x = af.data.tile(af.data.reorder(x), N_velocity, 1)
v = af.data.tile(v, 1, N_positions)

def calculate_n(f, v):
  deltav     = v[1, 0]-v[0, 0]
  value_of_n = af.sum(f, 0)*deltav
  return(value_of_n)

def calculate_T(f, v):
  deltav     = v[1, 0]-v[0, 0]
  value_of_T = af.sum(f*v**2, 0)*deltav 
  return(value_of_T)

def f_MB(x, v, f):
  n = calculate_n(f, v)
  n = af.data.tile(af.data.reorder(n), N_velocity, 1)
  T = calculate_T(f, v)
  T = af.data.tile(af.data.reorder(T), N_velocity, 1)

  f_MB = n*af.arith.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
           af.arith.exp(-mass_particle*v**2/(2*boltzmann_constant*T))
  
  return(f_MB)

def f_interp(dt, x, v, f):
  x_new     = x - v*dt
  step_size = af.sum(x[0, 1] - x[0, 0])
  f_interp  = af.randu(N_velocity, N_positions)

  for i in range(N_velocity):
      wall_indices = af.where((x_new[i, :]<0) + (x_new[i, :]>1.0))
      f_interp[i, :] = af.approx1(f[i, :], (x_new[i, :]/step_size))
      f_interp[i, wall_indices] = af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_walls))*\
                                  af.exp(-mass_particle*v[i, wall_indices]**2/(2*boltzmann_constant*T_walls))    

  return(f_interp)

f_initial = af.data.constant(0.005, N_velocity, N_positions)
f_current = f_initial

for time_index, t0 in enumerate(time):
  
  print("Computing For Time Index = ", time_index)
  print("Physical Time            = ", t0)

  # We shall split the Boltzmann-Equation and solve it:

  # In this step we are solving the collisionless equation
  fstar = f_interp(dt, x, v, f_current)
  
  # We turn off the term v(df/dx) for the following two steps
  f_intermediate = fstar - (dt/2)*(fstar          - f_MB(x, v, fstar))/tau
  f_new          = fstar - (dt)  *(f_intermediate - f_MB(x, v, fstar))/tau 
  f_current      = f_new