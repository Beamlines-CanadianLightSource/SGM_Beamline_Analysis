import json
import os
import numpy as np

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdd_calibration.json")

def load_calibration():
    """
    Loads the SDD calibration from the JSON file.
    Returns:
        dict: Calibration data or an empty dict if file doesn't exist.
    """
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load calibration file: {e}")
    return {}

def save_calibration(calib_data):
    """
    Saves the SDD calibration to the JSON file.
    """
    try:
        # Add a timestamp or metadata if needed
        import datetime
        calib_data["_metadata"] = {
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calib_data, f, indent=4)
        print(f"  -> [CALIBRATION] Saved to {CALIBRATION_FILE}")
        return True
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return False

def channel_to_energy(channel, gain, offset):
    """Converts channel to energy: E = Gain * Channel + Offset"""
    return gain * channel + offset

def energy_to_channel(energy, gain, offset):
    """Converts energy to channel: Channel = (Energy - Offset) / Gain"""
    if gain == 0:
        return 0
    return (energy - offset) / gain

def get_calibrated_bounds(energy_min, energy_max, sdd_id, calib_data):
    """
    Calculates the specific channel bounds for a given SDD.
    If no calibration exists for that SDD, returns default channel values.
    
    Args:
        energy_min (float): Minimum energy of ROI.
        energy_max (float): Maximum energy of ROI.
        sdd_id (str): Detector identifier (e.g., 'sdd1').
        calib_data (dict): Loaded calibration dictionary.
        
    Returns:
        tuple: (ch_start, ch_end)
    """
    if sdd_id in calib_data:
        gain = calib_data[sdd_id].get("gain", 1.0)
        offset = calib_data[sdd_id].get("offset", 0.0)
        
        ch_min = energy_to_channel(energy_min, gain, offset)
        ch_max = energy_to_channel(energy_max, gain, offset)
        
        # Ensure we return valid channel indices (integers)
        ch_start = int(max(0, np.floor(min(ch_min, ch_max))))
        ch_end = int(min(256, np.ceil(max(ch_min, ch_max))))
        return ch_start, ch_end
    
    # Fallback to assuming the user input energy was actually channels if no calibration
    return int(max(0, energy_min)), int(min(256, energy_max))

def calculate_calibration_params(channel_positions, energy_values):
    """
    Performs a linear fit to determine Gain and Offset.
    Energy = Gain * Channel + Offset
    """
    if len(channel_positions) < 2:
        return 1.0, 0.0 # Default
    
    # Fit: y = mx + c => Energy = Gain * Channel + Offset
    coeffs = np.polyfit(channel_positions, energy_values, 1)
    gain = coeffs[0]
    offset = coeffs[1]
    return float(gain), float(offset)
