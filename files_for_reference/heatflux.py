from __future__ import division
import numpy as np
import h5py
from scipy.integrate import nquad

N      = 32
dx     = 1./N
i      = np.arange(0, N, 1)
x      = (i + 0.5)*dx
tau    = 0.01
values = np.linspace(1.01,1.1,10)

count = 0
data = np.zeros(10)
for Tright in values:
  print("For temperature of right wall = ",Tright)

  def thetaInit(x):
    return(1+(Tright-1)*x)

  thetaInit = np.vectorize(thetaInit)

  def thetaWallLeft(x):
    return 1.0

  thetaWallLeft = np.vectorize(thetaWallLeft)

  def thetaWallRight(x):
    return Tright

  thetaWallRight = np.vectorize(thetaWallRight)


  def f0(v1, vPerp, v, x):

    if(x<=0):
    
      theta = thetaWallLeft(x)

      m = 1.
      k = 1.
       
      return 1 * (m/(2*np.pi*k*theta))**(3./2.) * np.exp(-m*v**2./(2.*k*theta))

    if(x>=1):
    
      theta = thetaWallRight(x)
      m = 1.
      k = 1.

      return 1 * (m/(2*np.pi*k*theta))**(3./2.) * np.exp(-m*v**2./(2.*k*theta))


    else:   

      theta = thetaInit(x)
      
      m = 1.
      k = 1.
         
      return 1 * (m/(2*np.pi*k*theta))**(3./2.) * np.exp(-m*v**2./(2.*k*theta))


  def f(v1, vPerp, v, x):
    
    if(x<=0):
    
      theta = thetaWallLeft(x)

      m = 1.
      k = 1.
       
      return 1 * (m/(2*np.pi*k*theta))**(3./2.) * np.exp(-m*v**2./(2.*k*theta))

    if(x>=1):
    
      theta = thetaWallRight(x)
      m = 1.
      k = 1.

      return 1 * (m/(2*np.pi*k*theta))**(3./2.) * np.exp(-m*v**2./(2.*k*theta))

    else:
      return (f0(v1, vPerp, v, x) - tau * v1 *(0.5*1e5) * (f0(v1, vPerp, v, x + 1e-5)-f0(v1, vPerp, v, x - 1e-5)))

  def heatfluxIntegrand(v1, vPerp, x):
    v = np.sqrt(v1**2. + vPerp**2.)
    integralMeasure = 2*np.pi*vPerp

    return (integralMeasure) * (v1*v**2) * f(v1, vPerp, v, x)

  print(thetaInit(x))

  solnVst = np.zeros(32)

  for gridPoint in range(N):
    print("gridPoint = ",gridPoint)

    integral = nquad(heatfluxIntegrand, [[-np.inf, np.inf], [0, np.inf]], args=[x[gridPoint]])
    solnVst[gridPoint]  = integral[0]   

  print(solnVst)
  print(np.sum(solnVst)/32)
  data[count] = np.sum(solnVst)/32
  count = count + 1
  h5f = h5py.File('analytical_'+str(Tright)+'.h5', 'w')
  h5f.create_dataset('sol', data=solnVst)
  h5f.close() 

"""
import pylab as pl
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20  
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8     
pl.rcParams['xtick.minor.size'] = 4     
pl.rcParams['xtick.major.pad']  = 8     
pl.rcParams['xtick.minor.pad']  = 8     
pl.rcParams['xtick.color']      = 'k'     
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'    

pl.rcParams['ytick.major.size'] = 8     
pl.rcParams['ytick.minor.size'] = 4     
pl.rcParams['ytick.major.pad']  = 8     
pl.rcParams['ytick.minor.pad']  = 8     
pl.rcParams['ytick.color']      = 'k'     
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'    

pl.plot(x,solnVst,label='Calculated')
pl.plot(x,thetaInit(x),label='Input')
pl.legend()
pl.xlabel('$x$')
pl.ylabel('$T$')
pl.savefig('plot.png')
"""

