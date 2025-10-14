import numpy as np
import time
import matplotlib.pyplot as plt
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.srs import SR830
from pyvisa import ResourceManager

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

# Set Keithley output voltage to 0 V and enable output
keithley.source_voltage = 0
keithley.output = True
time.sleep(1)  # Wait for stability

# Ramp down from Keithley voltage to -10 V in 0.1 V steps. Then wait 3 seconds
ramp_step = 0.1
ramp_delay = 0.1  # Adjust delay to suit your settling time

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

# Sweep parameters
step = 0.1
min_delay = 0.5
delay = max(min_delay, 5 * sr830.time_constant)  # seconds between steps

# Create sweep array from -10 V to 10 V
sweep_points = np.arange(-10, 10 + step, step)

data = []

print("Starting voltage sweep from -10 V to 10 V...")

for v in sweep_points:
    keithley.source_voltage = v
    time.sleep(delay)
    lockin_voltage = sr830.x
    print(f"Keithley V = {v:.3f} V, Lock-in V = {lockin_voltage:.6f} V")
    data.append((v, lockin_voltage))

print("Sweep complete.")

# Plot Keithley voltage vs lock-in voltage
voltages = [d[0] for d in data]
lockin_voltages = [d[1] for d in data]

plt.figure(figsize=(8,6))
plt.plot(voltages, lockin_voltages, marker='o', linestyle='-')
plt.xlabel("Keithley Output Voltage (V)")
plt.ylabel("Lock-in Measured Voltage (V)")
plt.title("Keithley Output vs Lock-in Measured Voltage")
plt.grid(True)
plt.savefig("keithley_vs_lockin.png", dpi=300)  # Save plot as PNG file
# plt.show()  # If environment supports it

# Sweep Keithley voltage back down to 0 V in steps
print("Sweeping Keithley output back down to 0 V...")
reverse_points = np.arange(voltages[-1], 0 - step, -step)
for v in reverse_points:
    keithley.source_voltage = v
    time.sleep(delay)

# Set lock-in amplitude to 0 V (turn off excitation)
sr830.amplitude = 0
time.sleep(0.5)

# Shutdown instruments
keithley.output = False
keithley.shutdown()

print("Instruments shut down. Done.")
