import pandas as pd
import numpy as np

def generate_monthly_telemetry_dataset(filename="medical_temperature_monthly.csv"):
    # 1. Generate 1-minute interval timestamps for 30 full days (43,200 total samples)
    start_date = '2026-09-01 00:00:00'
    end_date = '2026-09-30 23:59:00'
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')

    total_points = len(timestamps)
    
    # 2. Generate realistic medical temperature signal (°C)
    np.random.seed(42)
    base_temp = 4.0
    daily_sine_cycle = 1.5 * np.sin(np.linspace(0, 30 * 2 * np.pi, total_points))
    sensor_noise = np.random.normal(loc=0.0, scale=0.15, size=total_points)
    temperatures = np.round(base_temp + daily_sine_cycle + sensor_noise, 2)

    # 3. Add an afternoon thermal spike anomaly on Day 15
    spike_start = 14 * 1440 + 800  # Day 15 at ~01:20 PM
    temperatures[spike_start : spike_start + 60] += 5.5

    # 4. Create Initial DataFrame WITH sequential Sample IDs BEFORE dropping rows
    df = pd.DataFrame({
        'sample_id': np.arange(1, total_points + 1),  # IDs 1 to 43200
        'timestamp': timestamps,
        'temperature': temperatures,
        'node_id': 'MED_UNIT_01'
    })

    # 5. Simulate Outage Gap on Day 10 (Drop 45 rows WITHOUT re-indexing sample_id)
    gap_start_idx = 9 * 1440 + 360  # Day 10 at 06:00 AM
    df = df.drop(df.index[gap_start_idx : gap_start_idx + 45]).reset_index(drop=True)

    # DO NOT re-index sample_id! That preserves sample ID skips (e.g. 13320 -> 13366)

    # 6. Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Generated 1-Month CSV with true ID gaps: '{filename}' ({len(df):,} rows)")

if __name__ == "__main__":
    generate_monthly_telemetry_dataset()
