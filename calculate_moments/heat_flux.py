def calculate_heat_flux(f, v):
  deltav               = v[1, 0]-v[0, 0]
  value_of_temperature = np.sum(f*v**3, axis = 0)*deltav
  return(value_of_temperature)