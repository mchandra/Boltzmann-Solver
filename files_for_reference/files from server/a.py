import numpy as np
from scipy.interpolate import interp1d
import h5py
import arrayfire as af 
af.set_backend("cpu")

# Setting velocity and spatial grid points
N_positions = 6
ghost_zones = 3
N_velocity  = 11

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
v_max = 1.0

# Time Parameters for the simulation:
dt         = 0.05 # Size of the time-step
final_time = 3.0
time       = np.arange(dt, final_time, dt)

# Setting up the temperature parameters for the simulations:
T_initial = 1.5
T_walls   = 2.0

# Setting up of spatial and velocity grids:
x  = np.linspace(left_boundary, right_boundary, N_positions)
dx = x[1] - x[0]

# Obtaining the coordinates for the ghost-zones:
x_ghost_left  = np.linspace(-(ghost_zones)*dx + left_boundary, left_boundary - dx, ghost_zones)
x_ghost_right = np.linspace(right_boundary + dx, right_boundary + ghost_zones*dx , ghost_zones)

# Combining them to obtain the entire spatial grid
x  = np.concatenate([x_ghost_left, x, x_ghost_right])

# Obtaining the velocity grid
v  = np.linspace(-v_max, v_max, N_velocity)

# Conversion to allow for easy vectorization
x = x * np.ones([N_velocity , N_positions + 2*ghost_zones])
v = v * np.ones([N_positions + 2*ghost_zones, N_velocity])
v = np.transpose(v)

def calculate_density(f, v):
  deltav           = v[1, 0]-v[0, 0]
  value_of_density = np.sum(f, axis = 0)*deltav
  return(value_of_density)

def calculate_temperature(f, v):
  deltav               = v[1, 0]-v[0, 0]
  value_of_temperature = np.sum(f*v**2, axis = 0)*deltav
  return(value_of_temperature)

def f_MB(x, v, f):
  n = calculate_density(f, v) * np.ones((N_velocity, N_positions + 2*ghost_zones), dtype = np.float)
  T = calculate_temperature(f, v) * np.ones((N_velocity, N_positions + 2*ghost_zones), dtype = np.float)
  f_MB = n*np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
       np.exp(-mass_particle*v**2/(2*boltzmann_constant*T))
  return(f_MB)

# Function that return's the distribution function at the walls.
def f_walls(v):
  f_walls = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_walls))*\
        np.exp(-mass_particle*v**2/(2*boltzmann_constant*T_walls))
  return(f_walls)

def f_interp(dt, x, v, f):
  x_new     = x - v*dt
  f_interp  = np.zeros([N_velocity, N_positions + 2*ghost_zones])
  f_interp  = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_walls)) * \
              np.exp(-mass_particle*v**2/(2*boltzmann_constant*T_walls))
  print(af.to_array(x_new))
  for i in range(N_velocity):
    wall_indices   = np.where(  (x_new[i, ghost_zones:-ghost_zones]<=left_boundary) \
                  + (x_new[i, ghost_zones:-ghost_zones]>=right_boundary) \
                 )[0]
    all_indices    = np.arange(N_positions)
    indices_domain = np.delete(all_indices, wall_indices)

    f_interp[i, ghost_zones + indices_domain] = interp1d(x[i, :], f[i, :], 'cubic')(x_new[i, ghost_zones + indices_domain])
    f_interp[i, wall_indices + ghost_zones]   = f_walls(v[:, 0])[i]
  return f_interp

# Intializing the values for f
f_initial = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_initial)) * \
            np.exp(-mass_particle*v**2/(2*boltzmann_constant*T_initial))

# Declaring the indices of the ghost cells at the left and right walls
indices_left  = np.arange(ghost_zones)
indices_right = np.arange(N_positions + ghost_zones, N_positions + 2 * ghost_zones)
indices_walls = np.concatenate([indices_left, indices_right])

# Setting temperature at the walls:
for i in indices_walls:
    f_initial[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_walls)) * \
                      np.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_walls))

f_current = f_initial
time_data = np.zeros(time.size)

for time_index, t0 in enumerate(time):
  
  print("Computing For Time Index = ", time_index)
  print("Physical Time            = ", t0)

  # We shall split the Boltzmann-Equation and solve it:

  # In this step we are solving the collisionless equation
  fstar = f_interp(dt, x, v, f_current)
  fstar = af.to_array(fstar)
  print(fstar)
  break
  # We turn off the term v(df/dx) for the following two steps
  # f0             = f_MB(x, v, fstar)
  # f_intermediate = fstar - (dt/2)*(fstar          - f0)/tau
  # f_new          = fstar - (dt)  *(f_intermediate - f0)/tau
  f_new         = fstar
  indices_left  = np.arange(ghost_zones)
  indices_right = np.arange(N_positions + ghost_zones, N_positions + 2 * ghost_zones)
  indices_walls = np.concatenate([indices_left, indices_right])

  for i in indices_walls:
      f_new[:, i] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_walls)) * \
                    np.exp(-mass_particle*v[:, 0]**2/(2*boltzmann_constant*T_walls))
  
  f_current = f_new
  temp      = calculate_temperature(f_current[:, ghost_zones:-ghost_zones], v[:, ghost_zones:-ghost_zones])
  temp      = np.sum(temp)/N_positions
  time_data[time_index] = temp

h5f = h5py.File('cheng.h5', 'w')
h5f.create_dataset('temp_data',      data = time_data)
h5f.create_dataset('time',           data = time)
h5f.close()