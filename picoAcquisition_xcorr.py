import ctypes
import numpy as np
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
import matplotlib.pyplot as plt
import time as ti
from scipy.signal import welch, correlate, correlation_lags
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal.windows import blackman
import os

# Environment Variables
volt_scale=5
n_Bin=10
n_segments = 10
g=500*100 # Define amplifier gain
numAcquisitionsArray = [10, 20]
num_runs = 2
#Total number of waveforms = numAcquisition[i] * n_Bin * number of buffer segments (n_segments)

#######################

chandle = ctypes.c_int16()
status = {}

status["openunit"] = ps.ps4000OpenUnit(ctypes.byref(chandle))
assert_pico_ok(status["openunit"])

chARange = volt_scale
status["setChA"] = ps.ps4000SetChannel(chandle, 0, 1, 1, chARange)
assert_pico_ok(status["setChA"])

chBRange = volt_scale
status["setChB"] = ps.ps4000SetChannel(chandle, 1, 1, 1, chBRange)
assert_pico_ok(status["setChB"])

status["trigger"] = ps.ps4000SetSimpleTrigger(chandle, 1, 0, 1024, 2, 0, 1000)
assert_pico_ok(status["trigger"])

preTriggerSamples = 5000*n_Bin  ###splitting into n bins later
postTriggerSamples = 5000*n_Bin  ###splitting into n bins later
maxSamples = preTriggerSamples + postTriggerSamples

timebase = 4
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()
oversample = ctypes.c_int16(1)
status["getTimebase2"] = ps.ps4000GetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), oversample, ctypes.byref(returnedMaxSamples), 0)
assert_pico_ok(status["getTimebase2"])

# Functions
def xcorr_avg_mV(a_mv, b_mv, n_bins):
    A = np.asarray(a_mv, dtype=float)
    B = np.asarray(b_mv, dtype=float)
    A_splits = np.array_split(A, n_bins)
    B_splits = np.array_split(B, n_bins)
    acc = None
    for Ab, Bb in zip(A_splits, B_splits):
        c = correlate(Ab, Bb, mode="same", method="fft") / (len(Bb) * 1e6)
        acc = c if acc is None else (acc + c)
    return acc / n_bins

def acquire_one_batch(numAcquisitions, n_segments, n_Bin):

    average_cross_corr = None
    cmaxSamples_Time = ctypes.c_int32(np.int32(maxSamples/n_Bin))

    for i in range(numAcquisitions):
        nMaxSamples = ctypes.c_int32(0)
        status["setMemorySegments"] = ps.ps4000MemorySegments(chandle, n_segments, ctypes.byref(nMaxSamples))
        assert_pico_ok(status["setMemorySegments"])

        status["SetNoOfCaptures"] = ps.ps4000SetNoOfCaptures(chandle, n_segments)
        assert_pico_ok(status["SetNoOfCaptures"])
        
        buffersA = [(ctypes.c_int16 * maxSamples)() for _ in range(n_segments)]
        buffersB = [(ctypes.c_int16 * maxSamples)() for _ in range(n_segments)]

        for seg_idx in range(n_segments):
            status[f"setDataBufferA{seg_idx}"] = ps.ps4000SetDataBuffer(chandle, 0, ctypes.byref(buffersA[seg_idx]), maxSamples)

        for seg_idx in range(n_segments):
            status[f"setDataBufferB{seg_idx}"] = ps.ps4000SetDataBuffer(chandle, 0, ctypes.byref(buffersB[seg_idx]), maxSamples)

        status["runBlock"] = ps.ps4000RunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, oversample, None, 0, None, False)
        assert_pico_ok(status["runBlock"])

        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps4000IsReady(chandle, ctypes.byref(ready))

        overflow = (ctypes.c_int16 * n_segments)()
        cmaxSamples=ctypes.c_int32(np.int32(maxSamples))

        for seg_idx in range(n_segments):
            status["setDataBufferBulk"] = ps.ps4000SetDataBufferBulk(chandle, 0, buffersA[seg_idx], maxSamples, seg_idx)
        
        for seg_idx in range(n_segments):
            status["setDataBufferBulk"] = ps.ps4000SetDataBufferBulk(chandle, 1, buffersB[seg_idx], maxSamples, seg_idx)

        status["getValuesBulk"] = ps.ps4000GetValuesBulk(chandle, ctypes.byref(cmaxSamples), 0, n_segments - 1, ctypes.byref(overflow))
        assert_pico_ok(status["getValuesBulk"])

        maxADC = ctypes.c_int16(32767)

        adc2mVChA = [adc2mV(buffersA[i], chARange, maxADC) for i in range(n_segments)]
        adc2mVChB = [adc2mV(buffersB[i], chARange, maxADC) for i in range(n_segments)]

        A_mv = [np.array(ch) for ch in adc2mVChA]
        B_mv = [np.array(ch) for ch in adc2mVChB]

        seg_corrs = [xcorr_avg_mV(A_mv[k], B_mv[k], n_Bin) for k in range(n_segments)]
        cross_corr_avg = np.mean(np.stack(seg_corrs, axis=0), axis=0)
        
        if i == 0:
            average_cross_corr = cross_corr_avg
        else:
            average_cross_corr = [(x + y) for x, y in zip(average_cross_corr, cross_corr_avg)]
            
        print("Acquisition number:", i + 1)

    return average_cross_corr, cmaxSamples_Time

def one_sided_psd_from_corr(corr, fs, numAcquisitions):
    Sv_corr = np.absolute(fftshift(fft(corr)))
    Sv_corr=2*Sv_corr[Sv_corr.size//2:]
    Sv_corr=Sv_corr/fs
    Sv_corr=Sv_corr/(g * g * numAcquisitions) 
    return Sv_corr

def plot_psd(Sv_corr, title, label, fwelch, f_lo, f_hi):
    fig, ax = plt.subplots(figsize=(10,9), constrained_layout=True)
    ax.plot(fwelch, Sv_corr, alpha=1, label = label)
    ax.axhline(np.mean(Sv_corr[np.where((fwelch>f_lo) & (fwelch<f_hi))]), color='magenta', linestyle='--')
    ax.set_xlabel("frequency[Hz]")
    ax.set_ylabel("Sv[$V^2$/Hz]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e2,1e6)
    # ax.set_ylim(1e-21,1e-15)
    ax.axvline(f_hi, color='r', linestyle='--')
    ax.axvline(f_lo, color='r', linestyle='--')
    plt.title(title)
    ax.legend()

def run_psd_analysis(numAcquisitions):
    all_xcorrs = []

    for run in range(num_runs):
        avg_corr, cmaxSamples_Time = acquire_one_batch(numAcquisitions, n_segments, n_Bin)
        all_xcorrs.append(avg_corr)
        print(f"Completed run {run + 1}/{num_runs} for {numAcquisitions} acquisitions")

    # Create time/frequency axis
    time = np.linspace(0, ((cmaxSamples_Time.value) - 1) * timeIntervalns.value / 1e9, cmaxSamples_Time.value)
    T = (time.max() - time.min())
    N = np.size(time)
    fs = int(N / T)
    dt = 1 / fs
    fwelch = fftfreq(N, dt)
    fwelch = fwelch[:fwelch.size // 2]

    Sv_xcorr_runs = [one_sided_psd_from_corr(corr, fs, numAcquisitions) for corr in all_xcorrs]

    Sv_xcorr = np.mean(Sv_xcorr_runs, axis=0)

    # PSD printout
    f_lo = 100_000
    f_hi = 300_000
    print(f"\n===== Results for {numAcquisitions} acquisitions =====")
    print(f"Cross PSD  = {np.mean(Sv_xcorr[np.where((fwelch > f_lo) & (fwelch < f_hi))])} between {f_lo} and {f_hi} Hz")

    # Plot PSDs
    plot_psd(Sv_xcorr, f"FFT of {num_runs} Averaged Cross-Correlation ({numAcquisitions} Acq)", "cross correlation", fwelch, f_lo, f_hi)

    # Save data
    os.makedirs("outputs/Mass Acquisition Script", exist_ok=True)

    plt.savefig(f"outputs/Mass Acquisition Script/cryostat_12and11_gain_5e4_1e4avg_2Ms_10Ksa_20250911_{num_runs}avg_{numAcquisitions}acquisitions.png")
    np.savetxt(f'outputs/Mass Acquisition Script/cryostat_12and11_gain_5e4_1e4avg_2Ms_10Ksa_20250911_{num_runs}avg_{numAcquisitions}acquisitions.hdf5', [(x,y) for x,y in zip(fwelch,Sv_xcorr)], delimiter=',')
    
    print(f"Saved PSD data for {num_runs} averaging, {numAcquisitions} acquisitions.")

########################################################

# Runs everything
for numAcquisitions in numAcquisitionsArray:
    run_psd_analysis(numAcquisitions)


# Stop the scope
# handle = chandle
status["stop"] = ps.ps4000Stop(chandle)
assert_pico_ok(status["stop"])


#Close unit Disconnect the scope
#handle = chandle
status["close"] = ps.ps4000CloseUnit(chandle)
assert_pico_ok(status["close"])