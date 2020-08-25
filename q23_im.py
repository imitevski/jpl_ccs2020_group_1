import numpy as np 

##### Q2

# constants
e=2.718
rho_0 = 1.225 # kg/m^3
h = 10000 # meteras
N = 10400 # meters
Cp_air = 1.005*1000  #J/kg/K (~15deg)
dN = 0.6 # W/m2
dt=365*24*3600. #yr to second

m = rho_0 *(N - N * e**(-h/N))# mass
dT_atm = dN * dt / (m * Cp_air)

print(dT_atm,"K/yr, atmosphere")




##### Q3
Cp_ocn = 3850.  #J/kg/K (~15deg)
rho_ocn = 1023  #kg/m^3 (15deg)

dt_ocn_10 = dN * dt / (Cp_ocn * rho_ocn * 10)
dt_ocn_100 = dN * dt / (Cp_ocn * rho_ocn * 100)
dt_ocn_700 = dN * dt / (Cp_ocn * rho_ocn * 700)
dt_ocn_2000 = dN * dt / (Cp_ocn * rho_ocn * 2000)

print(dt_ocn_10,"K/yr for 10m")
print(dt_ocn_100,"K/yr for 100m")
print(dt_ocn_700,"K/yr for 700m")
print(dt_ocn_2000,"K/yr for 2000m")

'''
answers
2.3926246368530784 K/yr, atmosphere
0.4804204593060898 K/yr for 10m
0.04804204593060898 K/yr for 100m
0.0068631494186584254 K/yr for 700m
0.002402102296530449 K/yr for 2000m
'''
