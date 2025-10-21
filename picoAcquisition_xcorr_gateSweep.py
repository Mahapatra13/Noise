import ctypes
import numpy as np
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, fftfreq, fftshift
import os
from pyvisa import ResourceManager
from pyvisa.errors import VisaIOError
from pymeasure.instruments.keithley import Keithley2400
import matplotlib
import time as ti

# Backend setting for interactive plotting
matplotlib.use('TkAgg')  # or 'Qt5Agg' if Tk is not available

# Configure Keithley
KEITHLEY_ADDR = "GPIB0::25::INSTR"

rm = ResourceManager()
keithley = Keithley2400(rm.open_resource(KEITHLEY_ADDR))

keithley.reset()
keithley.use_front_terminals()
keithley.source_function = "voltage"
keithley.source_voltage_range = 20 # -10V to 10V
keithley.compliance_current = 0.01  # 10 mA limit
keithley.enable_source()

# Environment Variables
volt_scale=5
n_Bin=10
n_segments = 10
g=500*100 # Define amplifier gain
numAcquisitions = 50 # Sets number of acquisitions
num_runs = 1
#Total number of waveforms = numAcquisition[i] * n_Bin * number of buffer segments (n_segments)

ramp_step = 4 # Step size between -10 V and 10 V
ramp_delay = 0.1  # Adjust delay to suit your settling time
slow_ramp_step = 0.1 # The slow steps to take for avoiding transient current peaks

f_lo = 100_000
f_hi = 300_000

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

def run_psd_analysis(numAcquisitions, f_lo, f_hi, keithley_voltage):
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

    # Grab only the PSD segment in the desired frequency band
    band_mask = (fwelch > f_lo) & (fwelch < f_hi)
    band_freqs = fwelch[band_mask]
    band_psd = Sv_xcorr[band_mask]

    # PSD printout
    print(f"\n===== Results for {numAcquisitions} acquisitions =====")
    print(f"Cross PSD  = {np.mean(Sv_xcorr[np.where((fwelch > f_lo) & (fwelch < f_hi))])} between {f_lo} and {f_hi} Hz")

    # Plot PSDs
    plot_psd(Sv_xcorr, f"FFT of {num_runs} Averaged Cross-Correlation ({numAcquisitions} Acq) at {keithley_voltage}V Gate Voltage", "cross correlation", fwelch, f_lo, f_hi)

    # Save data
    os.makedirs("outputs/xcorrGateSweep", exist_ok=True)

    plt.savefig(f"outputs/xcorrGateSweep/cryostat_12and11_gain_5e4_1e4avg_2Ms_10Ksa_20250911_{num_runs}avg_{numAcquisitions}acquisitions_{keithley_voltage}V.png")
    np.savetxt(f'outputs/xcorrGateSweep/cryostat_12and11_gain_5e4_1e4avg_2Ms_10Ksa_20250911_{num_runs}avg_{numAcquisitions}acquisitions_{keithley_voltage}V.csv', [(x,y) for x,y in zip(fwelch,Sv_xcorr)], delimiter=',')
    
    print(f"Saved PSD data for {num_runs} averaging and {numAcquisitions} acquisitions at gate voltage {keithley_voltage}V .")
    
    return band_freqs, band_psd


def keithley_ramp(keithley, target_voltage, ramp_step=0.1, ramp_delay=0.1):
    """
    Smoothly ramps the Keithley source voltage from its current voltage
    to the target_voltage in steps of ramp_step with delay ramp_delay.

    Parameters:
    - keithley: Keithley instrument object
    - target_voltage: float, voltage to ramp to (Volts)
    - ramp_step: float, step size for voltage increments (Volts)
    - ramp_delay: float, delay between steps (seconds)
    """
    try:
        current_voltage = keithley.source_voltage
        print(f"Starting ramp from {current_voltage:.3f} V to {target_voltage:.3f} V")

        # Determine step direction (+ or -)
        step_direction = np.sign(target_voltage - current_voltage)
        if step_direction == 0:
            print("Already at target voltage.")
            return

        # Create ramp points array (exclude the last point, set it explicitly after)
        ramp_points = np.arange(current_voltage, target_voltage, step_direction * abs(ramp_step))
        
        for v in ramp_points:
            keithley.source_voltage = v
            ti.sleep(ramp_delay)

        # Ensure exact target voltage at the end
        keithley.source_voltage = target_voltage
        ti.sleep(ramp_delay)

        print(f"Reached target voltage: {target_voltage:.3f} V")

    except Exception as e:
        print(f"Error during ramp: {e}")

########################################################

try:
    # Ramps Keithley to -10V
    keithley_ramp(keithley, -10, slow_ramp_step, ramp_delay)

    gate_voltages = np.arange(-10, 10.1, ramp_step)
    print(np.shape(gate_voltages))
    psd_band_averages = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker='o')
    ax.set_xlabel("Gate Voltage (V$_{BG}$)")
    ax.set_ylabel(f"Avg PSD ({f_lo / 1000:.0f}–{f_hi / 1000:.0f} kHz)")
    ax.set_title("Gate Voltage vs Band-Averaged PSD")
    ax.grid(True)

    for v in gate_voltages:
        keithley_ramp(keithley, v, slow_ramp_step, ramp_delay)

        f_band, psd_band = run_psd_analysis(numAcquisitions, f_lo, f_hi, v)

        avg_psd = np.mean(psd_band)
        psd_band_averages.append(avg_psd)

        print(f"Gate V = {v:.2f} V | Band Avg PSD = {avg_psd:.3e}")

        # LIVE PLOTTING
        line.set_xdata(gate_voltages[:len(psd_band_averages)])
        line.set_ydata(psd_band_averages)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.savefig("outputs/xcorrGateSweep/gate_vs_psd.png", dpi=300)
    plt.show()

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected! Cleaning up and releasing hardware...")

finally:
    # Pico cleanup
    try:
        status["stop"] = ps.ps4000Stop(chandle)
        assert_pico_ok(status["stop"])
        print("PicoScope stopped.")
    except Exception as e:
        print(f"Failed to stop PicoScope: {e}")

    try:
        status["close"] = ps.ps4000CloseUnit(chandle)
        assert_pico_ok(status["close"])
        print("PicoScope connection closed.")
    except Exception as e:
        print(f"Failed to close PicoScope: {e}")

    # Keithley cleanup
    try:
        keithley_ramp(keithley, 0, slow_ramp_step, ramp_delay)
        keithley.disable_source()
        print("Keithley voltage set to 0 V and source disabled.")
    except Exception as e:
        print(f"Failed to safely disable Keithley: {e}")