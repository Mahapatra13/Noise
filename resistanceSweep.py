import numpy as np
import time
from pymeasure.instruments.keithley import Keithley2400
from pyvisa import ResourceManager

# Set your GPIB address here
GPIB_ADDRESS = "GPIB0::25::INSTR"

# Initialize the Keithley 2400
rm = ResourceManager()
keithley = Keithley2400(rm.open_resource(GPIB_ADDRESS))

# Configure the instrument
keithley.reset()
keithley.use_front_terminals()
keithley.source_function = "voltage"
keithley.source_voltage = 0
keithley.source_voltage_range = 20
keithley.compliance_current = 0.01  # 10 mA limit
keithley.enable_source()
keithley.measure_current()
keithley.auto_zero_enabled = True
keithley.output = True
time.sleep(0.5)

# Sweep parameters
step = 0.1
delay = 0.1  # seconds between steps

# Create full sweep array
sweep_points = np.concatenate([
    np.arange(0, 10.1, step),
    np.arange(10-step, -10.1, -step),
    np.arange(-10+step, 0.1, step)
])

# Store data
data = []

print("Starting voltage sweep...")

for v in sweep_points:
    keithley.source_voltage = v
    time.sleep(delay)
    current = keithley.current
    print(f"V = {v:.3f} V, I = {current:.6e} A")
    data.append((v, current))

print("Sweep complete.")

# Turn off output and close connection
keithley.output = False
keithley.shutdown()

# Optional: Save to file
with open("voltage_sweep_data.csv", "w") as f:
    f.write("Voltage (V),Current (A)\n")
    for v, i in data:
        f.write(f"{v},{i}\n")

print("Data saved to 'voltage_sweep_data.csv'")
