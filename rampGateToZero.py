import time
from pyvisa import ResourceManager
from pymeasure.instruments.keithley import Keithley2400

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

# Setup Keithley GPIB address
KEITHLEY_ADDR = "GPIB0::25::INSTR"

rm = ResourceManager()
keithley = Keithley2400(rm.open_resource(KEITHLEY_ADDR))

# Basic Keithley setup
keithley.reset()
keithley.use_front_terminals()
keithley.source_function = "voltage"
keithley.source_voltage_range = 20
keithley.compliance_current = 0.01  # 10 mA limit
keithley.enable_source()
keithley.output = True

# Call your imported ramp function
safe_ramp_to_zero(keithley, ramp_step=0.1, ramp_delay=0.1)

# Clean up Keithley output and shutdown
keithley.output = False
keithley.shutdown()
print("Ramp to zero complete and Keithley shutdown.")