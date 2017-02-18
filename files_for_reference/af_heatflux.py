import numpy as np
import arrayfire as af
af.set_backend("opencl")

# Setting velocity and spatial grid points
N_positions = 1001
ghost_zones = 3
N_velocity  = 1001

# Boundaries of domain
left_boundary  = 0
right_boundary = 1.0
length         = right_boundary - left_boundary

# Setting mass of the particle, boltzmann-constant
mass_particle      = 1.0
boltzmann_constant = 1.0

# Scattering time scale
tau   = 0.1
# Magnitude of maximum velocity
v_max = 10.0

# Time Parameters for the simulation:
dt         = 0.1*tau # Size of the time-step
final_time = 3.0
time       = np.arange(dt, final_time, dt)

# Setting up the temperature parameters for the simulations:
T_left   = 1.0
T_right  = 1.1
T_initial = 1.05

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
x  = af.to_array(x)
v  = af.to_array(v)

# Conversion to allow for easy vectorization
x = af.tile(x, 1, N_velocity)
v = af.tile(af.reorder(v), N_positions + 2*ghost_zones, 1)

def calculate_density(f, v):
    deltav           = af.sum(v[0, 1]-v[0, 0])
    value_of_density = af.sum(f, 1)*deltav
    return(value_of_density)

def calculate_temperature(f, v):
    deltav               = af.sum(v[0, 1]-v[0, 0])
    value_of_temperature = af.sum(f*v**2, 1)*deltav
    return(value_of_temperature)

def calculate_heatflux(f, v):
    deltav               = af.sum(v[0, 1]-v[0, 0])
    value_of_heatflux    = af.sum(f*v**3, 1)*deltav
    return(value_of_heatflux)

def f_MB(x, v, f):
    n = af.tile(calculate_density(f, v), 1, N_velocity)
    T = af.tile(calculate_temperature(f, v), 1, N_velocity)
    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*v**2/(2*boltzmann_constant*T))
    return(f_MB)

def f_interp(dt, x, v, f):
    x_new     = x - v*dt
    step_size = af.sum(x[1,0] - x[0,0])
    f_inter   = af.constant(0, N_positions + 2*ghost_zones, N_velocity)
    f_inter[ghost_zones:-ghost_zones,:] = af.approx1(f, x_new[ghost_zones:-ghost_zones,:]/step_size, af.INTERP.CUBIC)
    
    f_inter = af.Array.as_type(f_inter, af.Dtype.f64)
    
    f_left    = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_left))*\
                af.exp(-mass_particle*v**2/(2*boltzmann_constant*T_left))
    
    f_right   = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_right))*\
                af.exp(-mass_particle*v**2/(2*boltzmann_constant*T_right))
    
    f_inter[ghost_zones:-ghost_zones,:] = af.select(x_new[ghost_zones:-ghost_zones, :]<=left_boundary, \
                                                     f_left[ghost_zones:-ghost_zones,:], \
                                                     f_inter[ghost_zones:-ghost_zones,:]
                                                    )
    
    f_inter[ghost_zones:-ghost_zones,:] = af.select(x_new[ghost_zones:-ghost_zones, :]>=right_boundary, \
                                                     f_right[ghost_zones:-ghost_zones,:], \
                                                     f_inter[ghost_zones:-ghost_zones,:]
                                                    )
    
    return f_inter

# Intializing the values for f
f_initial = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_initial)) * \
            af.exp(-mass_particle*v**2/(2*boltzmann_constant*T_initial))

f_initial[:ghost_zones,:] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_left)) * \
                            af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_left))
    
f_initial[N_positions + ghost_zones:N_positions + 2 * ghost_zones, :] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_right)) * \
                                                                        af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_right))

f_current = f_initial

for time_index, t0 in enumerate(time):
    print("Computing For Time Index = ", time_index)
    #print("Physical Time            = ", t0)
    # We shall split the Boltzmann-Equation and solve it:
    # In this step we are solving the collisionless equation
    fstar = f_interp(dt, x, v, f_current)
    
    fstar[:ghost_zones,:] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_left)) * \
                            af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_left))
    
    fstar[N_positions + ghost_zones:N_positions + 2 * ghost_zones, :] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_right)) * \
                                                                        af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_right))

    # We turn off the term v(df/dx) for the following two steps
    f0             = f_MB(x, v, fstar)
    f_new          = f0 + (fstar - f0)*np.exp(-dt/tau)
    
    f_new[:ghost_zones,:] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_left)) * \
                            af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_left))
    
    f_new[N_positions + ghost_zones:N_positions + 2 * ghost_zones, :] = np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T_right)) * \
                                                                        af.exp(-mass_particle*v[:ghost_zones, :]**2/(2*boltzmann_constant*T_right))


    f_current = f_new

flux = calculate_heatflux(f_current, v)
avg  = af.sum(flux[3:-3])/(N_positions)
print(avg)
