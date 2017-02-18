import numpy as np
import h5py
from scipy.integrate import quad

# Setting up spatial parameters:
N_x  = 32  # No of spatial grid points
dx   = 1./N_x # Step size for spatial grid
i    = np.arange(0, N_x, 1)
x    = (i + 0.5)*dx

# Setting up temporal parameters:
N_t      = 100 # No of temporal grid points
t_final  = 3.0
t        = np.linspace(0, t_final, N_t)

rho_initial  = 1.0
temp_initial = 1.5
temp_walls   = 2.0
tau          = 1.0 # Scattering time scale

m = 1.0 # Mass of particle
k = 1.0 # Boltzmann constant

def rho_init(x):
  return rho_initial

rho_init = np.vectorize(rho_init)

def theta_init(x):
    return temp_initial

theta_init = np.vectorize(theta_init)

def theta_walls(x):
    return temp_walls

theta_walls = np.vectorize(theta_walls)

def f0(v, x):
  # Setting the temperature corresponding to that of the wall beyond boundaries
  if(x<=0):
    theta = theta_walls(x)
    return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))
  
  if(x>=1):
    theta = theta_walls(x)
    return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))

  else:   
    rho   = rho_init(x)
    theta = theta_init(x)     
    return rho * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))

def f(v, x, t):
    
    if(x<=0):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))

    if(x>=1):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))
                

    else:
        return f0(v, x - v*t)

def pressure_integrand(v, x, t):
    integral_measure = 1.0
    return integral_measure * v**2. * f(v, x, t)

soln = np.zeros([N_x,N_t])

for grid_point in range(N_x):
    for n in range(N_t):
        print( "n = ", n,"grid_point = ",grid_point)

        integral             = quad(pressure_integrand, -np.inf, np.inf, args=(x[grid_point], t[n]))
        soln[grid_point][n]  = integral[0]

print(soln)
h5f = h5py.File('analytical.h5', 'w')
h5f.create_dataset('time', data=t)
h5f.create_dataset('soln', data=soln)
h5f.close() 