import numpy as np
import time
import matplotlib.pyplot as plt
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.srs import SR830
from pyvisa import ResourceManager
from pyvisa.errors import VisaIOError
import matplotlib
import os

matplotlib.use('TkAgg')  # or 'Qt5Agg' if Tk is not available

# Set GPIB addresses
KEITHLEY_ADDR = "GPIB0::25::INSTR"
SR830_ADDR = "GPIB0::8::INSTR"

# Initialize Instruments
rm = ResourceManager()
keithley = Keithley2400(rm.open_resource(KEITHLEY_ADDR))
sr830 = SR830(rm.open_resource(SR830_ADDR))

# Configure the Keithley 2400
keithley.reset()
keithley.use_front_terminals()
keithley.source_function = "voltage"
keithley.source_voltage_range = 20
keithley.compliance_current = 0.01  # 10 mA limit
keithley.enable_source()

# Ramp Keithley output voltage to 0 V and enable output
ramp_step = 0.1
ramp_delay = 0.1  # Adjust delay to suit your settling time

# Sweep parameters
step = 0.1
min_delay = 0.5

# Create data containers
voltages = []
lockin_voltages = []

# Helper Functions
def safe_ramp_to_zero(keithley, ramp_step=0.1, ramp_delay=0.1):
    try:
        # Try to read current voltage
        try:
            current_voltage = keithley.source_voltage
        except Exception as e:
            print(f"Failed to read current Keithley voltage: {e}")
            return  # Cannot ramp if we don't know the voltage

        if abs(current_voltage) <= 1e-3:
            print("Keithley already at ~0 V. No ramp needed.")
            keithley.source_voltage = 0
            return

        print(f"Ramping Keithley voltage from {current_voltage:.3f} V to 0 V...")
        ramp_direction = -np.sign(current_voltage)
        ramp_points = np.arange(current_voltage, 0 + ramp_direction * ramp_step, ramp_direction * ramp_step)

        for v in ramp_points:
            try:
                keithley.source_voltage = v
                time.sleep(ramp_delay)
            except Exception as e:
                print(f"Failed to set Keithley voltage to {v:.3f} V: {e}")
                break  # Stop ramp if communication fails

        # Ensure exactly 0 V at the end
        try:
            keithley.source_voltage = 0
        except Exception as e:
            print(f"Failed to set final Keithley voltage to 0 V: {e}")

        time.sleep(0.5)

    except Exception as e:
        print(f"Unexpected error during ramp to 0 V: {e}")

try:
    # Configure the Keithley 2400
    keithley.reset()
    keithley.use_front_terminals()
    keithley.source_function = "voltage"
    keithley.source_voltage_range = 20
    keithley.compliance_current = 0.01  # 10 mA limit
    keithley.enable_source()

    current_voltage = keithley.source_voltage
    keithley.output = True
    time.sleep(0.5)

    # Ramp to 0 V if not already near zero
    safe_ramp_to_zero(keithley, ramp_step, ramp_delay)

    # Ramp down from Keithley voltage to -10 V in 0.1 V steps. Then wait 3 seconds
    print("Ramping Keithley voltage down from 0 V to -10 V...")
    for v in np.arange(0, -10 - ramp_step, -ramp_step):
        keithley.source_voltage = v
        time.sleep(ramp_delay)

    time.sleep(3)

    # Configure the SR830 Lock-In after Keithley stabilized at -10 V
    sr830.reference_source = 'Internal'   # Use internal reference
    sr830.frequency = 13                  # Set frequency
    sr830.sine_voltage = 5.0              # Voltage output amplitude
    sr830.input_configuration = 'a-b'     # Use A-B differential input
    sr830.input_coupling = 'AC'           # AC coupling
    sr830.time_constant = 0.03            # 30 ms time constant
    sr830.sensitivity = 0.1               # 100 mV sensitivity (adjust if needed)
    time.sleep(0.5)                       # Wait to settle

    # Sweep delay based on lock-in time constant
    delay = max(min_delay, 5 * sr830.time_constant)  # seconds between steps

    # Create sweep array from -10 V to 10 V
    sweep_points = np.arange(-10, 10 + step, step)

    data = []

    # Enable interactive plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], marker='o', linestyle='-')
    ax.set_xlabel("V$_{BG}$ (Keithley)")
    ax.set_ylabel("V$_{AB}$ (Lock-In)")
    ax.set_title("V$_{BG}$ vs V$_{AB}$")
    ax.grid(True)
    plt.show()

    print("Starting voltage sweep from -10 V to 10 V...")

    for v in sweep_points:
        keithley.source_voltage = v
        time.sleep(delay)
        lockin_voltage = sr830.x
        print(f"Keithley V = {v:.3f} V, Lock-in V = {lockin_voltage:.6f} V")

        voltages.append(v)
        lockin_voltages.append(lockin_voltage)

        # Update the plot data
        line.set_data(voltages, lockin_voltages)
        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Sweep complete.")
    plt.ioff()  # Turn off interactive mode

    # Save plot
    save_path = os.path.abspath("V$_{BG}$ vs V$_{AB}$.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    # Sweep Keithley voltage back down to 0 V in steps
    print("Sweeping Keithley output back down to 0 V...")
    reverse_points = np.arange(voltages[-1], 0 - step, -step)
    for v in reverse_points:
        keithley.source_voltage = v
        time.sleep(delay)

    # Ensure final setpoint is 0 V
    keithley.source_voltage = 0
    time.sleep(0.5)

# Handle common runtime issues gracefully
except KeyboardInterrupt:
    print("\nMeasurement interrupted by user. Cleaning up...")
except VisaIOError as e:
    print(f"\nVISA communication error: {e}")
except Exception as e:
    print(f"\nUnexpected error occurred: {e}")

# Always ensure instruments are safely shut down
finally:
    try:
        sr830.sine_voltage = 0  # Set lock-in amplitude to 0 V (turn off excitation)
    except Exception as e:
        print(f"Could not set lock-in amplitude to 0: {e}")
    time.sleep(0.5)

    try:
        safe_ramp_to_zero(keithley, ramp_step, ramp_delay)
        keithley.output = False
        keithley.shutdown()
    except Exception as e:
        print(f"Error shutting down Keithley: {e}")

    print("Instruments shut down. Done.")