import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =====================================================================
# 1. FILE LOADER & GAP FILL ENGINE
# =====================================================================
class TelemetryProcessor:
    def __init__(self, file_path, target_period_sec=2.0):
        self.file_path = file_path
        self.target_period_sec = target_period_sec

    def load_and_reconstruct_gaps(self):
        df = pd.read_csv(self.file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        processed_rows = []
        for i in range(len(df) - 1):
            curr_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            row_dict = curr_row.to_dict()
            row_dict['data_source'] = 'LIVE'
            row_dict['zoh_temp'] = curr_row['temperature']
            processed_rows.append(row_dict)
            
            time_delta = (next_row['timestamp'] - curr_row['timestamp']).total_seconds()
            
            if time_delta > (1.5 * self.target_period_sec):
                missing_steps = int(time_delta // self.target_period_sec) - 1
                
                temp_start = curr_row['temperature']
                temp_end = next_row['temperature']
                temp_step = (temp_end - temp_start) / (missing_steps + 1)
                
                for step in range(1, missing_steps + 1):
                    interp_time = curr_row['timestamp'] + pd.Timedelta(seconds=step * self.target_period_sec)
                    zoh_temp = temp_start
                    interpolated_temp = round(temp_start + (step * temp_step), 2)
                    
                    processed_rows.append({
                        'sample_id': f"~{int(curr_row['sample_id']) + step}",
                        'timestamp': interp_time,
                        'temperature': interpolated_temp,
                        'zoh_temp': zoh_temp,
                        'node_id': curr_row['node_id'],
                        'data_source': 'PREDICTED_INTERPOLATED'
                    })
                    
        last_row = df.iloc[-1].to_dict()
        last_row['data_source'] = 'LIVE'
        last_row['zoh_temp'] = last_row['temperature']
        processed_rows.append(last_row)
        
        return pd.DataFrame(processed_rows)


# =====================================================================
# 2. FEATURE ENGINEERING ANALYTICS ENGINE
# =====================================================================
class TelemetryAnalyticsEngine:
    @staticmethod
    def process_dataframe(df, temp_col='temperature', time_col='timestamp', temp_limit=8.0, ambient_temp=22.0):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        df['rolling_median'] = df[temp_col].rolling(window=5, min_periods=1).median()
        df['ema_baseline'] = df[temp_col].ewm(alpha=0.3, adjust=False).mean()

        dt = df[time_col].diff().dt.total_seconds().fillna(1.0).replace(0, 1.0)
        dtemp = df[temp_col].diff().fillna(0.0)
        df['rate_of_rise'] = dtemp / dt
        df['delta_t_ambient'] = df[temp_col] - ambient_temp

        rolling_std = df[temp_col].rolling(window=5, min_periods=1).std().fillna(0.0).replace(0, 1e-6)
        rolling_mean = df[temp_col].rolling(window=5, min_periods=1).mean()
        df['z_score'] = (df[temp_col] - rolling_mean) / rolling_std
        df['is_anomaly'] = df['z_score'].abs() > 3.0

        df['temp_excursion'] = np.maximum(0, df[temp_col] - temp_limit)
        df['cumulative_stress'] = (df['temp_excursion'] * dt).cumsum()

        return df


# =====================================================================
# 3. 30-DAY HOURLY AGGREGATION & MONTHLY REPORT ENGINE
# =====================================================================
def save_hierarchical_rollups(df, db_name="medical_telemetry_archive.db", output_csv="hourly_temperature_averages.csv", output_monthly="monthly_temperature_summary.xlsx"):
    temp_df = df.copy()
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df['hour'] = temp_df['timestamp'].dt.hour
    temp_df['date'] = temp_df['timestamp'].dt.date

    # Baseline hourly continuous rollup
    hourly_df = temp_df.set_index('timestamp').resample('1h').agg(
        hourly_avg_temperature=('temperature', 'mean'),
        min_temperature=('temperature', 'min'),
        max_temperature=('temperature', 'max'),
        sample_count=('temperature', 'count'),
        anomaly_count=('is_anomaly', 'sum')
    ).dropna().reset_index()

    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
    hourly_df['hour_window'] = hourly_df['timestamp'].dt.strftime('%H:00 - %H:59')
    hourly_df['hourly_avg_temperature'] = hourly_df['hourly_avg_temperature'].round(2)

    export_columns = ['hour', 'hour_window', 'hourly_avg_temperature', 'min_temperature', 'max_temperature', 'sample_count', 'anomaly_count']
    try:
        hourly_df[export_columns].to_csv(output_csv, index=False)
    except PermissionError:
        pass

    # Monthly aggregation across 30 days
    hourly_30day_agg = temp_df.groupby('hour').agg(
        avg_30day_temp=('temperature', 'mean'),
        min_30day_temp=('temperature', 'min'),
        max_30day_temp=('temperature', 'max')
    ).reset_index()

    hourly_30day_agg['avg_30day_temp'] = hourly_30day_agg['avg_30day_temp'].round(2)
    hourly_30day_agg['min_30day_temp'] = hourly_30day_agg['min_30day_temp'].round(2)
    hourly_30day_agg['max_30day_temp'] = hourly_30day_agg['max_30day_temp'].round(2)

    max_hourly_avg_val = hourly_30day_agg['avg_30day_temp'].max()
    min_hourly_avg_val = hourly_30day_agg['avg_30day_temp'].min()

    monthly_records = []
    for idx, row in hourly_30day_agg.iterrows():
        hr = int(row['hour'])
        avg_val = row['avg_30day_temp']

        if avg_val == max_hourly_avg_val:
            status_tag = "MAXIMUM HOURLY AVERAGE (MONTHLY PEAK)"
        elif avg_val == min_hourly_avg_val:
            status_tag = "MINIMUM HOURLY AVERAGE (MONTHLY TROUGH)"
        else:
            status_tag = "NORMAL"

        monthly_records.append({
            'Hour Slot': f"Hour {hr:02d} ({hr:02d}:00 - {hr:02d}:59)",
            '30-Day Average Temp (°C)': avg_val,
            '30-Day Min Temp (°C)': row['min_30day_temp'],
            '30-Day Max Temp (°C)': row['max_30day_temp'],
            'Hourly Average Status': status_tag
        })

    monthly_report_df = pd.DataFrame(monthly_records)

    daily_averages = temp_df.groupby('date')['temperature'].mean()
    peak_single_day_avg = round(daily_averages.max(), 2)
    hottest_day_str = str(daily_averages.idxmax())

    global_monthly_min = round(temp_df['temperature'].min(), 2)
    global_monthly_max = round(temp_df['temperature'].max(), 2)

    summary_row = pd.DataFrame([{
        'Hour Slot': '--- MONTHLY SUMMARY TOTALS ---',
        '30-Day Average Temp (°C)': f"Peak Single-Day Avg: {peak_single_day_avg} °C ({hottest_day_str})",
        '30-Day Min Temp (°C)': global_monthly_min,
        '30-Day Max Temp (°C)': global_monthly_max,
        'Hourly Average Status': 'SUMMARY TOTALS'
    }])

    final_monthly_df = pd.concat([monthly_report_df, summary_row], ignore_index=True)

    try:
        final_monthly_df.to_excel(output_monthly, index=False, sheet_name="30Day_Hourly_Summary")
    except Exception:
        fallback_csv = output_monthly.replace('.xlsx', '.csv')
        final_monthly_df.to_csv(fallback_csv, index=False)

    with sqlite3.connect(db_name) as conn:
        hourly_df.to_sql('hourly_telemetry', conn, if_exists='replace', index=False)
        monthly_report_df.to_sql('monthly_hourly_profile', conn, if_exists='replace', index=False)

    return hourly_df, final_monthly_df


# =====================================================================
# 4. SECONDARY DASHBOARD: MONTHLY SUMMARY DASHBOARD (REAL-TIME)
# =====================================================================
class MonthlySummaryDashboard(tk.Toplevel):
    """Separate Real-Time Dashboard Window for the Monthly Temperature Summary."""
    def __init__(self, parent, get_monthly_df_fn):
        super().__init__(parent)
        self.title("Monthly Temperature Analytics — Dedicated Real-Time Dashboard")
        self.geometry("950x650")
        self.get_monthly_df_fn = get_monthly_df_fn

        self._build_ui()
        self.start_auto_refresh()

    def _build_ui(self):
        # Header Banner
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(top_frame, text="📈 Monthly Temperature Summary (Live Stream)", font=("Segoe UI", 13, "bold"), foreground="darkblue").pack(side="left")
        
        self.lbl_updated = ttk.Label(top_frame, text="Status: Live Auto-Sync", font=("Segoe UI", 10, "italic"), foreground="green")
        self.lbl_updated.pack(side="right")

        # Summary Metric Cards Frame
        self.cards_frame = ttk.LabelFrame(self, text=" Monthly Key Aggregations ")
        self.cards_frame.pack(fill="x", padx=15, pady=5)

        self.lbl_peak_avg = ttk.Label(self.cards_frame, text="Peak Day Avg: -- °C", font=("Segoe UI", 10, "bold"), foreground="navy")
        self.lbl_peak_avg.pack(side="left", padx=15, pady=8)

        self.lbl_monthly_max = ttk.Label(self.cards_frame, text="Monthly Max: -- °C", font=("Segoe UI", 10, "bold"), foreground="red")
        self.lbl_monthly_max.pack(side="left", padx=15, pady=8)

        self.lbl_monthly_min = ttk.Label(self.cards_frame, text="Monthly Min: -- °C", font=("Segoe UI", 10, "bold"), foreground="blue")
        self.lbl_monthly_min.pack(side="left", padx=15, pady=8)

        # Table Section
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=15, pady=10)

        cols = ("slot", "avg", "min", "max", "status")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings")

        self.table.heading("slot", text="Hour Slot")
        self.table.heading("avg", text="30-Day Avg (°C)")
        self.table.heading("min", text="30-Day Min (°C)")
        self.table.heading("max", text="30-Day Max (°C)")
        self.table.heading("status", text="Hourly Average Status")

        self.table.column("slot", width=180, anchor="center")
        self.table.column("avg", width=220, anchor="center")
        self.table.column("min", width=120, anchor="center")
        self.table.column("max", width=120, anchor="center")
        self.table.column("status", width=250, anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)

        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def refresh_data(self):
        """Fetches latest monthly summary data and updates table and metric cards."""
        monthly_df = self.get_monthly_df_fn()
        
        for item in self.table.get_children():
            self.table.delete(item)

        for _, row in monthly_df.iterrows():
            slot = str(row['Hour Slot'])
            avg_val = str(row['30-Day Average Temp (°C)'])
            min_val = str(row['30-Day Min Temp (°C)'])
            max_val = str(row['30-Day Max Temp (°C)'])
            status = str(row['Hourly Average Status'])

            self.table.insert("", "end", values=(slot, avg_val, min_val, max_val, status))

        # Update Banner Metric Cards from Summary Row
        summary_row = monthly_df[monthly_df['Hourly Average Status'] == 'SUMMARY TOTALS']
        if not summary_row.empty:
            s_data = summary_row.iloc[0]
            self.lbl_peak_avg.config(text=f"Peak Single-Day Avg: {s_data['30-Day Average Temp (°C)']}")
            self.lbl_monthly_max.config(text=f"Monthly Max: {s_data['30-Day Max Temp (°C)']} °C")
            self.lbl_monthly_min.config(text=f"Monthly Min: {s_data['30-Day Min Temp (°C)']} °C")

        self.lbl_updated.config(text=f"Last Synced: {datetime.now().strftime('%H:%M:%S')}")

    def start_auto_refresh(self):
        """Auto-refreshes the secondary dashboard every 2 seconds."""
        self.refresh_data()
        self.after(2000, self.start_auto_refresh)


# =====================================================================
# 5. SECONDARY DASHBOARD: HOURLY AVERAGES DASHBOARD
# =====================================================================
class HourlyAveragesWindow(tk.Toplevel):
    def __init__(self, parent, hourly_df):
        super().__init__(parent)
        self.title("Hourly Averages Dashboard — Live Rollup View")
        self.geometry("750x550")
        self.hourly_df = hourly_df

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(top_frame, text="📊 Live Hourly Averages Tracker (24 Hours)", font=("Segoe UI", 12, "bold"), foreground="navy").pack(side="left")

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)

        cols = ("hour", "window", "avg", "min", "max", "samples")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings")

        self.table.heading("hour", text="Hour #")
        self.table.heading("window", text="Time Window")
        self.table.heading("avg", text="Hourly Avg (°C)")
        self.table.heading("min", text="Min Temp (°C)")
        self.table.heading("max", text="Max Temp (°C)")
        self.table.heading("samples", text="Samples Recorded")

        self.table.column("hour", width=70, anchor="center")
        self.table.column("window", width=140, anchor="center")
        self.table.column("avg", width=110, anchor="center")
        self.table.column("min", width=100, anchor="center")
        self.table.column("max", width=100, anchor="center")
        self.table.column("samples", width=120, anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)

        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for _, row in self.hourly_df.head(24).iterrows():
            hr = f"Hour {int(row['hour']):02d}"
            wnd = row['hour_window']
            avg_val = f"{row['hourly_avg_temperature']:.2f}"
            min_val = f"{row['min_temperature']:.2f}"
            max_val = f"{row['max_temperature']:.2f}"
            smpl = str(int(row['sample_count']))

            self.table.insert("", "end", values=(hr, wnd, avg_val, min_val, max_val, smpl))

        final_daily_avg = round(self.hourly_df['hourly_avg_temperature'].mean(), 2)

        bottom_frame = ttk.LabelFrame(self, text=" Final 24-Hour Consolidated Summary ")
        bottom_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(bottom_frame, text="Consolidated Daily Average (From 24 Hourly Points):", font=("Segoe UI", 10)).pack(side="left", padx=10, pady=8)
        ttk.Label(bottom_frame, text=f"{final_daily_avg} °C", font=("Segoe UI", 11, "bold"), foreground="darkgreen").pack(side="left", padx=5)


# =====================================================================
# 6. MAIN GUI RECEIVER DASHBOARD
# =====================================================================
class CombinedTelemetryDashboard(tk.Tk):
    def __init__(self, csv_file="medical_temperature_raw.csv", db_name="integrated_telemetry.db"):
        super().__init__()
        self.title("HART HardwAIre Receiver — Comprehensive Telemetry & Analytics Dashboard")
        self.geometry("1600x900")

        self.csv_file = csv_file
        self.db_name = db_name
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.outage_simulated = True
        self.recovery_in_progress = False
        
        self.is_calibrated = False
        self.calibrated_ticks = 23981

        self._init_sqlite()

        if os.path.exists(self.csv_file):
            processor = TelemetryProcessor(self.csv_file)
            reconstructed_df = processor.load_and_reconstruct_gaps()
            self.df_raw = TelemetryAnalyticsEngine.process_dataframe(reconstructed_df)
        else:
            fallback = self._generate_fallback_dataset()
            self.df_raw = TelemetryAnalyticsEngine.process_dataframe(fallback)

        self.hourly_df, self.monthly_summary_df = save_hierarchical_rollups(self.df_raw)

        self.base_window = self.df_raw.iloc[11200:11350].copy().reset_index(drop=True) if len(self.df_raw) > 11350 else self.df_raw.copy()
        
        self.split_idx = 80
        self.live_df = self.base_window.iloc[:self.split_idx].copy()
        self.live_df['source'] = 'LIVE'
        
        self.processed_df = TelemetryAnalyticsEngine.process_dataframe(self.live_df)

        self._build_top_bar()
        self._build_analytics_summary_bar()
        self._build_main_panes()
        self.update_dashboard()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                measured_time TEXT,
                raw_value INTEGER,
                temperature_c REAL,
                source TEXT
            );
            """)
            conn.commit()

    def _generate_fallback_dataset(self):
        timestamps = [datetime.now() - timedelta(seconds=i*2) for i in range(150, 0, -1)]
        return pd.DataFrame({
            'sample_id': np.arange(1, 151),
            'timestamp': timestamps,
            'temperature': np.round(36.0 + np.sin(np.linspace(0, 10, 150)), 2),
            'node_id': 'MED_UNIT_01',
            'data_source': 'LIVE'
        })

    def _build_top_bar(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(top_frame, text="Receiver Status:", font=("Segoe UI", 11, "bold")).pack(side="left")
        self.status_label = ttk.Label(top_frame, text="SIMULATED LINK LOSS - AUTO RECOVERY ARMED", font=("Segoe UI", 11), foreground="red")
        self.status_label.pack(side="left", padx=5)

        ttk.Label(top_frame, text=f"Session: {self.session_id}", font=("Segoe UI", 10)).pack(side="left", padx=10)

        self.clock_label = ttk.Label(top_frame, text="Clock: not calibrated", font=("Segoe UI", 10, "italic"), foreground="orange")
        self.clock_label.pack(side="left", padx=10)

        ttk.Label(top_frame, text="View Range:", font=("Segoe UI", 10, "bold")).pack(side="right", padx=(10, 2))
        self.view_var = tk.StringVar(value="Latest 150 Samples")
        self.view_combo = ttk.Combobox(top_frame, textvariable=self.view_var, values=["Latest 150 Samples", "All Datapoints"], state="readonly", width=25)
        self.view_combo.pack(side="right", padx=5)
        self.view_combo.bind("<<ComboboxSelected>>", lambda e: self.update_dashboard())

        # Navigation & Action Buttons
        self.btn_monthly_win = ttk.Button(top_frame, text="📈 Open Monthly Summary Dashboard", command=self.open_monthly_summary_dashboard)
        self.btn_monthly_win.pack(side="right", padx=3)

        self.btn_hourly_win = ttk.Button(top_frame, text="📊 Open Hourly Averages Window", command=self.open_hourly_averages_window)
        self.btn_hourly_win.pack(side="right", padx=3)

        self.btn_recall = ttk.Button(top_frame, text="🔄 Recall Missed Datasets", command=self.trigger_selective_recovery)
        self.btn_recall.pack(side="right", padx=3)

        self.btn_revert = ttk.Button(top_frame, text="⚡ Revert to Outage Graph", command=self.revert_to_outage_state)
        self.btn_revert.pack(side="right", padx=3)

        self.btn_calibrate = ttk.Button(top_frame, text="⏱️ Calibrate Clock", command=self.toggle_clock_calibration)
        self.btn_calibrate.pack(side="right", padx=3)

    def open_monthly_summary_dashboard(self):
        """Launches the dedicated Monthly Summary Dashboard window."""
        MonthlySummaryDashboard(self, lambda: self.monthly_summary_df)

    def open_hourly_averages_window(self):
        HourlyAveragesWindow(self, self.hourly_df)

    def _build_analytics_summary_bar(self):
        avg_frame = ttk.LabelFrame(self, text=" Analytics Summary & System Metrics ")
        avg_frame.pack(fill="x", padx=10, pady=4)

        lbl_style = ("Segoe UI", 9)
        val_style = ("Segoe UI", 9, "bold")

        daily_avg = round(self.df_raw['temperature'].mean(), 2)
        peak_temp = round(self.df_raw['temperature'].max(), 2)
        tot_anomalies = self.df_raw['is_anomaly'].sum()
        total_stress = round(self.df_raw['cumulative_stress'].iloc[-1], 1)

        ttk.Label(avg_frame, text="Daily Avg:", font=lbl_style).pack(side="left", padx=(8, 2))
        ttk.Label(avg_frame, text=f"{daily_avg} °C", font=val_style, foreground="green").pack(side="left", padx=(0, 15))

        ttk.Label(avg_frame, text="Peak Temp:", font=lbl_style).pack(side="left", padx=(8, 2))
        ttk.Label(avg_frame, text=f"{peak_temp} °C", font=val_style, foreground="red").pack(side="left", padx=(0, 15))

        ttk.Label(avg_frame, text="EMA Baseline (Latest):", font=lbl_style).pack(side="left", padx=(8, 2))
        latest_ema = round(self.df_raw['ema_baseline'].iloc[-1], 2)
        ttk.Label(avg_frame, text=f"{latest_ema} °C", font=val_style, foreground="purple").pack(side="left", padx=(0, 15))

        ttk.Label(avg_frame, text="Anomalies Flagged:", font=lbl_style).pack(side="left", padx=(8, 2))
        ttk.Label(avg_frame, text=f"{tot_anomalies}", font=val_style, foreground="darkred").pack(side="left", padx=(0, 15))

        ttk.Label(avg_frame, text="Cumulative Stress:", font=lbl_style).pack(side="left", padx=(8, 2))
        ttk.Label(avg_frame, text=f"{total_stress} °C·s", font=val_style, foreground="brown").pack(side="left", padx=(0, 10))

    def _build_main_panes(self):
        main_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill="both", expand=True, padx=10, pady=5)

        table_frame = ttk.Frame(main_pane)
        main_pane.add(table_frame, weight=1)

        cols = ("sample", "time", "temp", "ema", "ror", "zscore", "status")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings")
        
        self.table.heading("sample", text="Sample ID")
        self.table.heading("time", text="Time")
        self.table.heading("temp", text="Temp (°C)")
        self.table.heading("ema", text="EMA Baseline")
        self.table.heading("ror", text="Rate of Rise")
        self.table.heading("zscore", text="Z-Score")
        self.table.heading("status", text="Status")

        self.table.column("sample", width=70, anchor="center")
        self.table.column("time", width=110, anchor="center")
        self.table.column("temp", width=75, anchor="center")
        self.table.column("ema", width=85, anchor="center")
        self.table.column("ror", width=85, anchor="center")
        self.table.column("zscore", width=75, anchor="center")
        self.table.column("status", width=95, anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        graph_frame = ttk.Frame(main_pane)
        main_pane.add(graph_frame, weight=2)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.plot_temp = self.figure.add_subplot(211)
        self.plot_zscore = self.figure.add_subplot(212, sharex=self.plot_temp)

        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def toggle_clock_calibration(self):
        self.is_calibrated = not self.is_calibrated
        if self.is_calibrated:
            self.clock_label.config(text=f"Clock: calibrated ({self.calibrated_ticks} ticks)", foreground="green")
            if not self.outage_simulated:
                self.status_label.config(text="CLOCK CALIBRATED — TIMESTAMPS SYNCHRONIZED", foreground="blue")
        else:
            self.clock_label.config(text="Clock: not calibrated", foreground="orange")
        self.update_dashboard()

    def trigger_selective_recovery(self):
        if not self.outage_simulated or self.recovery_in_progress:
            return

        self.status_label.config(text="RECOVERY IN PROGRESS — RECONSTRUCTING HISTORY...", foreground="green")
        self.recovery_in_progress = True

        recovered_slice = self.base_window.iloc[self.split_idx:].copy()
        recovered_slice['source'] = 'RECOVERED'

        self.live_df = pd.concat([self.live_df, recovered_slice]).drop_duplicates(subset=['sample_id']).reset_index(drop=True)
        self.processed_df = TelemetryAnalyticsEngine.process_dataframe(self.live_df)

        self.outage_simulated = False
        self.recovery_in_progress = False
        self.status_label.config(text="CONNECTED — HISTORY FULLY RECOVERED", foreground="blue")
        
        self.update_dashboard()

    def revert_to_outage_state(self):
        self.live_df = self.base_window.iloc[:self.split_idx].copy()
        self.live_df['source'] = 'LIVE'
        self.processed_df = TelemetryAnalyticsEngine.process_dataframe(self.live_df)
        self.outage_simulated = True
        self.status_label.config(text="SIMULATED LINK LOSS - AUTO RECOVERY ARMED", foreground="red")
        self.update_dashboard()

    def update_dashboard(self):
        for item in self.table.get_children():
            self.table.delete(item)

        view_mode = self.view_var.get()
        display_df = self.df_raw if view_mode != "Latest 150 Samples" else self.processed_df

        for _, row in display_df.head(100).iterrows():
            sid = str(row['sample_id'])
            t_str = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')
            temp_c = f"{row['temperature']:.2f}"
            ema_val = f"{row['ema_baseline']:.2f}"
            ror_val = f"{row['rate_of_rise']:.3f}"
            z_val = f"{row['z_score']:.2f}"
            
            if not self.is_calibrated and row.get('source', '') == 'LIVE':
                status = "CALIBRATION"
            else:
                status = row.get('source', row.get('data_source', 'LIVE'))

            self.table.insert("", "end", values=(sid, t_str, temp_c, ema_val, ror_val, z_val, status))

        self.plot_temp.clear()
        self.plot_zscore.clear()

        if view_mode != "Latest 150 Samples":
            self.plot_temp.plot(self.df_raw['timestamp'], self.df_raw['temperature'], color='#1f77b4', linewidth=0.8, label='Temperature')
            self.plot_temp.plot(self.df_raw['timestamp'], self.df_raw['ema_baseline'], color='purple', linestyle='--', linewidth=1.0, label='EMA Baseline')
            self.plot_zscore.plot(self.df_raw['timestamp'], self.df_raw['z_score'], color='orange', linewidth=0.8, label='Z-Score')
            self.plot_temp.set_title("Full Dataset — Continuous Signal with EMA & Anomaly Tracking", fontsize=10, fontweight='bold')
        else:
            df_live = self.processed_df[self.processed_df.get('source', 'LIVE') == 'LIVE']
            x_live = np.arange(len(df_live))
            
            self.plot_temp.plot(x_live, df_live['temperature'], color='#1f77b4', linewidth=1.5, marker='.', label='Temperature')
            self.plot_temp.plot(x_live, df_live['ema_baseline'], color='purple', linestyle='--', linewidth=1.2, label='EMA Baseline')
            
            df_rec = self.processed_df[self.processed_df.get('source', '') == 'RECOVERED']
            if not df_rec.empty:
                x_rec = np.arange(len(df_live), len(df_live) + len(df_rec))
                self.plot_temp.scatter(x_rec, df_rec['temperature'], color='green', marker='o', s=30, label='Recovered', zorder=5)

            if self.outage_simulated:
                last_temp = df_live.iloc[-1]['temperature']
                x_pred = np.arange(len(df_live), len(df_live) + 40)
                y_pred = np.full(40, last_temp) + np.linspace(0, 0.08, 40)
                self.plot_temp.scatter(x_pred, y_pred, color='#d62728', marker='x', s=25, label='Predicted (provisional)', zorder=4)

            self.plot_zscore.plot(x_live, df_live['z_score'], color='orange', linewidth=1.2, marker='.', label='Z-Score Anomaly Indicator')
            self.plot_temp.set_title("HART Temperature & EMA Signal Tracking", fontsize=10, fontweight='bold')

        self.plot_temp.set_ylabel("Temp (°C)", fontsize=9)
        self.plot_temp.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        self.plot_temp.legend(loc='upper right', fontsize=8)

        self.plot_zscore.axhline(3.0, color='red', linestyle=':', label='Upper Anomaly Bound (+3σ)')
        self.plot_zscore.axhline(-3.0, color='red', linestyle=':', label='Lower Anomaly Bound (-3σ)')
        self.plot_zscore.set_ylabel("Z-Score (σ)", fontsize=9)
        self.plot_zscore.set_xlabel("Sample Timeline", fontsize=9)
        self.plot_zscore.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        self.plot_zscore.legend(loc='upper right', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()


# =====================================================================
# MAIN EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    CSV_FILE = "medical_temperature_monthly.csv" if os.path.exists("medical_temperature_monthly.csv") else "medical_temperature_raw.csv"

    if os.path.exists(CSV_FILE):
        processor = TelemetryProcessor(CSV_FILE)
        reconstructed_df = processor.load_and_reconstruct_gaps()
        analytics_df = TelemetryAnalyticsEngine.process_dataframe(reconstructed_df)
        save_hierarchical_rollups(analytics_df)

    app = CombinedTelemetryDashboard(csv_file=CSV_FILE)
    app.mainloop()
