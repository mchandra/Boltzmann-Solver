import numpy as np
from scipy.optimize import root

def solve(right, left):
	
	v_right   = right[0]
	rho_right = right[1]
	e_right   = right[2]

	v_left    = left[0]
	rho_left  = left[1]
	e_left    = left[2]

	fluxContinuityLeft  = flux(v_left, rho_left, e_left)[0]
	fluxContinuityRight = flux(v_right, rho_right, e_right)[0]
	
	fluxMomentumLeft  = flux(v_left, rho_left, e_left)[1]
	fluxMomentumRight = flux(v_right, rho_right, e_right)[1]

	fluxEnergyLeft  = flux(v_left, rho_left, e_left)[2]
	fluxEnergyRight = flux(v_right, rho_right, e_right)[2]

	fluxContinuityDifference = fluxContinuityLeft - fluxContinuityRight
	fluxMomentumDifference   = fluxMomentumLeft   - fluxMomentumRight
	fluxEnergyDifference     = fluxEnergyLeft     - fluxEnergyRight
	
	return([fluxContinuityDifference, fluxMomentumDifference, fluxEnergyDifference])

# This function gives out flux values for (v,rho,e)

def flux(v,rho,e):
	fluxContinuity = rho*v
	fluxMomentum   = (2/3)*rho*e + rho*v**2  
	fluxEnergy     = ((5/3)*rho*e + 0.5*rho*v**2)*v
	return([fluxContinuity, fluxMomentum, fluxEnergy])

leftValues        = np.array([1,1,1])
rightValuesGuess  = np.array([1,0.1,0.1])

sol = root(solve, rightValuesGuess, leftValues, method='lm')

print("The right side values of (v,rho,e) are:",sol.x)
print("The right side fluxes are:", flux(sol.x[0], sol.x[1], sol.x[2]))
print("The left side values of (v,rho,e) are:", leftValues)
print("The left side fluxes are:",flux(leftValues[0], leftValues[1], leftValues[2]))
