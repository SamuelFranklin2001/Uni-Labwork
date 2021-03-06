import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

#data = pd.read_excel("nitrogen.xls", sheet_name="100ml", names=("time","voltage"), usecols=(0,1), skiprows=1)
#data = pd.read_excel("nitrogen.xls", sheet_name="80ml", names=("time","voltage"), usecols=(0,1), skiprows=1)
#data = pd.read_excel("nitrogen.xls", sheet_name="60ml", names=("time","voltage"), usecols=(0,1), skiprows=1)
#data = pd.read_excel("nitrogen.xls", sheet_name="40ml", names=("time","voltage"), usecols=(0,1), skiprows=1)
data = pd.read_excel("nitrogen.xls", sheet_name="20ml", names=("time","voltage"), usecols=(0,1), skiprows=1)

emf=np.array(data.voltage)
time=np.array(data.time)

voltage=np.array(data.voltage)
time=np.array(data.time)
FTfreq = np.fft.rfftfreq(len(time),time[1]-time[0])
FT = np.fft.rfft(voltage)
aFT = np.abs(FT)
new_aFT = list() # new list for amplitudes greater than a certain value
for i in range(0, len(aFT)): # loop to find values greater than min
    if aFT[i] > 2.0:
        new_aFT.append(aFT[i])
    else:
        new_aFT.append(0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(time,emf)
ax1.set_ylabel("emf")
ax1.set_xlabel("time")

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(FTfreq, aFT)
ax2.set_xlabel("freq")

plt.show()

#peak ang. freq
for i in range(len(new_aFT)):
    if new_aFT[i]==max(new_aFT):
        maxAmp = new_aFT[i]
        maxAmpFreq = FTfreq[i]
print("Peak angular freg ", maxAmpFreq, "rads/s")

damped=max(aFT)
hdamped=damped/(2**0.5)
hdampedlist=[]
for i in range(0,301):
    hdampedlist.append(hdamped)
ax2.plot(FTfreq, hdampedlist)

def solve(f,x):
    s = np.sign(f)
    z = np.where(s == 0)[0]
    if z:
        return z
    else:
        s = s[0:-1] + s[1:]
        z = np.where(s == 0)[0]
        return z

def interp(f,x,z):
    if __name__ == '__main__':
        m = (f[z+1]-f[z])/(x[z+1]-x[z])
        return x[z] - f[z]/m
f = aFT - hdampedlist
z = solve(f, FTfreq)
ans = interp(f, FTfreq, z)

print(ans)

plt.show()