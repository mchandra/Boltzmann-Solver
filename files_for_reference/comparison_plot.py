import numpy as np
import pylab as pl
import h5py
from scipy.integrate import nquad
from sympy import integrate, Symbol, exp

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

h5f = h5py.File('analytical.h5', 'r')
sol = h5f['soln'][:]
time = h5f['time'][:]
h5f.close()
h5f = h5py.File('cheng.h5', 'r')
sol2 = h5f['temp_data'][:]
time2 = h5f['time'][:]
h5f.close()

print(sol)
new=np.zeros(time.size)
for i in range(1,31):
    new+=sol[i][:]
new=new/30
pl.plot(time, np.abs(new),'b', label='$\mathrm{Analytical}$')
pl.plot(time2, np.abs(sol2),'r', label='$\mathrm{Numerical}$')
pl.title('$\mathrm{Analytical/Numerical}$ $\mathrm{Solution}$')

pl.legend(loc = 'best')
pl.savefig('cheng-knorr_vs_analytic.png')
