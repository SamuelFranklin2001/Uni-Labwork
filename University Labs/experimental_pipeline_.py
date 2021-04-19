import matplotlib.pyplot as plt ### plotting things
import numpy as np ## one of python's main maths packages
import pandas as pd ## for reading in our data
from scipy.optimize import curve_fit ## for fitting a line to our data

#Defining constant variables:

P_0 = 101100
g = 9.81
mass = 106.68e-3
d = 34.16e-3
a = np.pi * (d/2)**2
P = P_0 + ((mass*g)/a)

constant = mass/(P*a**2)
print(constant)

#load data

Volumes = [100, 90, 80, 70, 60, 50, 40, 30, 20,10]

#time, voltage = np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/100mla.txt", skiprows= 2, unpack = True)
#V = 100e-6
#V_index = 0

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/90mlc.txt", skiprows= 2, unpack = True)
#V = 90e-6 # volume in m^3
#V_index = 1

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/80mla.txt", skiprows= 2, unpack = True)#V = 80e-6
#V = 80e-6
#V_index = 2

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/70mla.txt", skiprows= 2, unpack = True)#V = 70e-6 # volume in m^3
#V = 70e-6
#V_index = 3

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/60mla.txt", skiprows= 2, unpack = True)#V = 60e-6
#V = 60e-6
#V_index = 4

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/50mla.txt", skiprows= 2, unpack = True)#V = 50e-6 # volume in m^3
#V = 50e-6
#V_index = 5

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/40mla.txt", skiprows= 2, unpack = True)#V = 50e-6 # volume in m^3
#V = 40e-6 # volume in m^3
#V_index = 6

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/30mla.txt", skiprows= 2, unpack = True)
#V = 30e-6 # volume in m^3
#V_index = 7

#time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/20mla.txt", skiprows= 2, unpack = True)#V = 50e-6 # volume in m^3
#V = 20e-6
#V_index = 8

time, voltage= np.loadtxt("/Users/Hannah/Documents/sem2 project/nitrogendata/10mla.txt", skiprows= 2, unpack = True)#V = 50e-6 # volume in m^3
V = 10e-6 # volume in m^3
V_index = 9


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(time, voltage)
plt.xlabel('Time / s')
plt.ylabel('Emf / V')
plt.plot()
plt.show()

#%%

print("EMF vs time data for " + str(Volumes[V_index]) + "ml.")

#%%


cut_off_point = float(input("cut off point: "))
print(cut_off_point)

time_clean = []
voltage_clean = []

for i in range(0, len(time)):
    if time[i] >= 0 and time[i] <= cut_off_point:
        time_clean.append(time[i])
        voltage_clean.append(voltage[i])

#print(len(time_clean))
#print(len(voltage_clean))

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(time_clean, voltage_clean)
plt.ylabel("EMF / V")
plt.xlabel("Time / s")
plt.plot()
#plt.show()

#%%
time = np.arange(0, cut_off_point, 0.0001)

def fit(time, A, b, omega_star, phi):
    beta = b/(2*mass)
    exp_term = A*np.exp(-beta*time)
    cos = np.cos(omega_star*time + phi)

    return exp_term*(cos)

popt, pcov = curve_fit(fit, time_clean, voltage_clean)
#print("A =", popt[0], "+/-", pcov[0,0]**0.5)
#print("b =", popt[1], "+/-", pcov[1,1]**0.5)
#print("omega_star =", popt[2], "+/-", pcov[2,2]**0.5)
#print("phi =", popt[3], "+/-", pcov[3,3]**0.5)

A = popt[0]
b = popt[1]
omega_star = popt[2]
phi = popt[3]

omega_star_error = pcov[2, 2]**0.5
#print('Error in omega star is:{}'.format(omega_star_error))

fig = plt.figure(figsize=(11,7))
plt.plot(time, fit(time, A, b, omega_star, phi), color = "blue", linewidth = 2)
plt.plot(time_clean, voltage_clean, marker = 'o', color = 'black', ls = '--', markersize = 3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Time / s", size = 15)
plt.ylabel("EMF / mV", size = 15)

plt.plot()
plt.show()

# %%

# now it's time to fit the data to a damped wave


# define a function.

time = np.arange(0, cut_off_point, 0.0001)


def fit(time, A, b, omega_star, phi):
    beta = b / (2 * mass)
    exp_term = -A * np.exp(-beta * time)
    sin = omega_star * np.sin(time * omega_star + phi)
    cos = beta * np.cos(omega_star * time + phi)

    return emf_constant * exp_term * (sin + cos)


popt, pcov = curve_fit(fit, time_clean, voltage_clean)
print("A =", popt[0], "+/-", pcov[0, 0] ** 0.5)
print("b =", popt[1], "+/-", pcov[1, 1] ** 0.5)
print("omega_star =", popt[2], "+/-", pcov[2, 2] ** 0.5)
print("phi =", popt[3], "+/-", pcov[3, 3] ** 0.5)

A = popt[0]
b = popt[1]
omega_star = popt[2]
phi = popt[3]
omega_star_err = pcov[2, 2] ** 0.5

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111)
plt.plot(time, fit(time, A, b, omega_star, phi), color="black", linewidth=2)
plt.plot(time_clean, voltage_clean, marker='o', color='black', ls='none', markersize=3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Time / s", size=15)
plt.ylabel("EMF / V", size=15)
plt.tick_params(direction='in',  # I like 'in', could be 'out' or both 'inout'
                length=7,  # A reasonable length
                bottom='on',  # I want ticks on the bottom axes
                left='on',
                top='on',
                right='on')
plt.plot()
plt.show()

#%%
print(b)
def beta():
    return b/(2*mass)
print("Damping coefficient, beta, is {} for {}ml".format(beta(), str(Volumes[V_index])))
def omega_0():
    return np.sqrt(omega_star**2 + beta()**2)
print("Omega_0 is {} for {}ml".format(omega_0(), str(Volumes[V_index])))

#%%
fit = pd.read_excel("/Users/Hannah/Documents/sem2 project/nitrogendata/final.xls", names=('volume', 'inverse_volumes', 'omega_star', 'omega0'))

inverse_volumes = []
omega_star_fit = []
omega_0_fit = []

for i in range(0, len(fit.inverse_volumes)):
    if fit.inverse_volumes[i] >= 0:
        inverse_volumes.append(fit.inverse_volumes[i])
        omega_star_fit.append(fit.omega_star[i])
        omega_0_fit.append(fit.omega0[i])

Omega_squared = [i ** 2 for i in omega_0_fit]

print(omega_star_fit)
plt.plot(inverse_volumes, Omega_squared, 'o--')
plt.xlabel('inverse volume')
plt.ylabel('omega squared')
plt.show()

# read these into lists.


# %%

# fit a straight line

def line(x, m, c):
    return m * x + c


popt, pcov = curve_fit(line, inverse_volumes, Omega_squared)
print("m =", popt[0])
print("c =", popt[1])
gradient = popt[0]
intercept = popt[1]

x_array = np.arange(0, 30000, 1)
plt.plot(inverse_volumes, Omega_squared, 'o')
plt.plot(x_array, line(x_array, gradient, intercept))
plt.xlabel(r'Inverse Volume', size=15)
plt.ylabel('Natural Frequency squared', size=15)
plt.show()


# %%

#gamma = gradient * constant


## alternative methods include FT and Q factor -- investigate??