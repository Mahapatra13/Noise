import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import welch, correlate, correlation_lags
from scipy.signal.windows import hamming, hann
from scipy.integrate import trapezoid
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('C:/Users/pbsii/Desktop/Py Visa codes/Noise/nikhils_plot_style.mplstyle')
import seaborn as sns

files = ['C:/Users/pbsii/Desktop/Py Visa codes/Noise/Pico_NF_SR560_100_A.csv',
         'C:/Users/pbsii/Desktop/Py Visa codes/Noise/Pico_NF_SR560_100_A.csv']
         
df_1=pd.read_csv(files[0],header=None,sep=",")
df_1.rename(columns={0:"time"}, inplace=True)

df_2=pd.read_csv(files[1],header=None,sep=",")
df_2.rename(columns={0:"time"}, inplace=True)

#define gain
g=5000

####################################### correlation function ###############################################
def correlation_nik_vectorized(V1, V2):
    assert V1.size==V2.size
    n = V1.size
    lags = list(range(0,n,1))
    V2doubled = np.concatenate((V2,V2))
    cor = np.zeros(n)
    for lag in lags:
        cor[lag] = V1.dot(V2doubled[lag:lag+n])
    cor = cor/n
    return cor

############################Time axis#######################################
T = (df_1["time"].max()-df_1["time"].min()) # seconds
N = df_1.shape[0]
fs = int(N/T) # Hz
dt = 1/fs
time = df_1["time"].to_numpy() - df_1["time"].min()
print(f"Sampling frequency is {1/dt} Hz")
fwelch = fftfreq(N,dt)
fwelch = fwelch[:fwelch.size//2]

########### mean after doing cross correlation ##################################

Sv_list = np.zeros((df_1.shape[0]//2, df_1.shape[1]-1)) # for crosscorrelation
for i in range(1,df_1.shape[1]):
    V1 = (df_1[i].to_numpy())*1e-3
    V1=V1-V1.mean()
    # V2=V1 # auto correlation for comparison
    V2 = (df_2[i].to_numpy())*1e-3
    V2=V2-V2.mean()
    # V1=V2 # auto correlation for comparison
    ###############PSD from cross-correlation####################################
    #ac = correlation_nik_vectorized(V1,V2)
    ac=(correlate(V1,V2,mode='same', method='fft'))/V1.size
    Sv = np.absolute(fftshift(fft(ac)))
    Sv = 2*Sv[Sv.size//2:]
    Sv = Sv/fs
    Sv_list[:,i-1] = Sv

Sv_mean = np.mean(Sv_list, axis=1)/(g*g)

########### cross correlation after taking the mean ##################################

V3=np.array([])
V4=np.array([])

for i in range(1,df_1.shape[1]):
    Vx = (df_1[i].to_numpy())*1e-3
    Vx=Vx-Vx.mean()
    # V2=V1 # auto correlation for comparison
    Vy = (df_2[i].to_numpy())*1e-3
    Vy=Vy-Vy.mean()
    V3=np.append(V3,Vx)
    V4=np.append(V4,Vy)
    

V3=V3.reshape(N,(df_1.shape[1]-1), order='F')
V4=V4.reshape(N,(df_1.shape[1]-1), order='F')
V3=np.mean(V3, axis=1)
V4=np.mean(V4, axis=1)
#ac2 = correlation_nik_vectorized(V3,V4)
ac2=(correlate(V3,V4,mode='same', method='fft'))/V3.size

Sv_corr = np.absolute(fftshift(fft(ac2)))
Sv_corr = 2*Sv_corr[Sv_corr.size//2:]
Sv_corr = Sv_corr/fs
Sv_corr= Sv_corr/(g*g)




###########plotting###########################################################

f_lo = 10_000 # Hz
f_hi = 30_000 # Hz
label = f"{files[0].split('/')[-1].split('.')[0].split('_')[0]}"

print(f"{label} = {np.mean(Sv_mean[np.where((fwelch>f_lo) & (fwelch<f_hi))])} between {f_lo} and {f_hi} Hz")
fig, ax = plt.subplots(figsize=(10,9), constrained_layout=True)
ax.plot(fwelch, Sv_mean, alpha=1, label = f"{label}")
ax.axhline(np.mean(Sv_mean[np.where((fwelch>f_lo) & (fwelch<f_hi))]), color='magenta', linestyle='--')
ax.set_xlabel("frequency[Hz]")
ax.set_ylabel("Sv[$V^2$/Hz]")
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(1e-18,1e-9)
# ax.set_xlim(1e0,fwelch.max())
ax.axvline(f_hi, color='g', linestyle='--')
ax.axvline(f_lo, color='g', linestyle='--')
# ax.axhline(johnson_noise_psd(300,500_000), color='r', linestyle='--', label="Johnson noise")
ax.legend()
plt.show()

