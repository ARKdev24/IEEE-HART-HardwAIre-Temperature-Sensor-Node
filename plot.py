import pandas as pd
import matplotlib.pyplot as plt

def plot_existing_dataset():
    try:
        df = pd.read_csv('sensor_data.csv')
    except FileNotFoundError:
        print("Error: 'sensor_data.csv' not found in this folder.")
        return

    # Check for TimeSec (case-sensitive based on your CSV structure)
    if 'TimeSec' in df.columns:
        df['time_hours'] = df['TimeSec'] / 3600.0
    elif 'time_sec' in df.columns:
        df['time_hours'] = df['time_sec'] / 3600.0
    else:
        print("Error: Could not find time column in CSV! Columns found:", df.columns.tolist())
        return

    temp_col = 'Temperature' if 'Temperature' in df.columns else 'temp'

    # --- PLOTTING SECTION ---
    plt.figure(figsize=(14, 6))
    plt.plot(df['time_hours'], df[temp_col], label='Sensor Temp', color='blue', linewidth=1.5)

    # Background State Shading
    plt.axhspan(4.5, 7.5, color='green', alpha=0.1, label='State 0 (Normal)')
    plt.axhspan(7.5, 7.6, color='yellow', alpha=0.3, label='State 1 (Hold 20m)')
    plt.axhspan(7.6, 7.7, color='orange', alpha=0.3, label='State 2 (Hold 20m)')
    plt.axhspan(7.7, 8.2, color='red', alpha=0.2, label='States 3-5 (Chaos)')

    # Failure Threshold Line
    plt.axhline(y=8.0, color='black', linestyle='--', linewidth=1.5, label='Failure Threshold (8.0°C)')

    plt.xlim(71.0, 80.0)
    plt.ylim(4.5, 8.2)

    # Legend Positioned in the Top Right Corner (Keeping this change as requested)
    plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)

    plt.xlabel("Time (Hours)", fontsize=11)
    plt.ylabel("Temperature (°C)", fontsize=11)
    plt.title("Temperature Degradation Profile (Day 4 Anomaly)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    print("Rendering plot successfully...")
    plt.show()

if __name__ == '__main__':
    plot_existing_dataset()
