#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "sqlite3.h"

using namespace std;

struct DataPoint {
    double time_sec;
    double temp;
};

// Calculates OLS slope over the active window
double calculateOLSSlope(const deque<DataPoint>& window) {
    if (window.size() < 2) return 0.0;

    double sum_t = 0, sum_y = 0, sum_ty = 0, sum_t2 = 0;
    int n = window.size();
    double t_start = window.front().time_sec; 

    for (const auto& pt : window) {
        double t_i = (pt.time_sec - t_start) / 60.0; // relative time in minutes
        double y_i = pt.temp;                        // Temperature
        
        sum_t += t_i;
        sum_y += y_i;
        sum_ty += t_i * y_i;
        sum_t2 += t_i * t_i;
    }

    double denominator = (n * sum_t2) - (sum_t * sum_t);
    if (denominator == 0) return 0.0;

    return ((n * sum_ty) - (sum_t * sum_y)) / denominator;
}

// Stability check looking at the recent tail-end
bool isTemperatureStable(const deque<DataPoint>& window) {
    if (window.size() < 3) return false;
    
    double latest_time = window.back().time_sec;
    double sub_window_sec = 30 * 60; // 30-min tail check
    
    double min_temp = window.front().temp;
    double max_temp = window.front().temp;
    int points_counted = 0;

    for (auto it = window.rbegin(); it != window.rend(); ++it) {
        if (latest_time - it->time_sec <= sub_window_sec) {
            min_temp = min(min_temp, it->temp);
            max_temp = max(max_temp, it->temp);
            points_counted++;
        } else {
            break; 
        }
    }
    
    if (points_counted < 3) return false;
    return (max_temp - min_temp) < 0.03;
}

int main() {
    // --- 1. INITIALIZE SQLITE DATABASE ---
    sqlite3* db;
    char* errMessage = 0;
    
    int rc = sqlite3_open("sensor_logs.db", &db);
    if (rc) {
        cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
        return 1;
    }

    string sql_create_table = 
        "CREATE TABLE IF NOT EXISTS predictions (" \
        "id INTEGER PRIMARY KEY AUTOINCREMENT," \
        "time_sec REAL," \
        "temperature REAL," \
        "state INTEGER," \
        "ttf_mins REAL," \
        "status TEXT);";
    
    sqlite3_exec(db, sql_create_table.c_str(), 0, 0, &errMessage);

    // --- 2. OPEN DATASET ---
    ifstream file("sensor_data.csv");
    if (!file.is_open()) {
        cout << "Error: Could not open sensor_data.csv. Run dataset generator first!\n";
        sqlite3_close(db);
        return 1;
    }

    string line;
    getline(file, line); // Skip header

    deque<DataPoint> window;
    const double WINDOW_SEC = 30 * 60; // 30-minute rolling window
    const double FAILURE_THRESHOLD = 8.0;

    cout << "--- Predictive Maintenance OLS Engine (SQLite Logging Active) ---\n";
    cout << "Time(h) | Temp(C) | Slope(C/min) | Time to Failure | Status\n";
    cout << "--------------------------------------------------------------------------\n";

    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        DataPoint pt;
        int state, sample_rate;

        getline(ss, val, ','); pt.time_sec = stod(val);
        getline(ss, val, ','); pt.temp = stod(val);
        getline(ss, val, ','); state = stoi(val);
        getline(ss, val, ','); sample_rate = stoi(val);

        window.push_back(pt);

        while (!window.empty() && (pt.time_sec - window.front().time_sec > WINDOW_SEC)) {
            window.pop_front();
        }

        if (state >= 1 && state <= 5) {
            double current_hours = pt.time_sec / 3600.0;
            
            if (current_hours >= 72.5 && current_hours <= 74.5) {
                double m = calculateOLSSlope(window);
                bool stable_plateau = isTemperatureStable(window);
                
                double ttf_mins = 0.0;
                string status = "";

                cout << fixed << setprecision(2) 
                     << current_hours << "h  | " 
                     << pt.temp << "    | "
                     << setw(10) << m << " | ";

                if (m <= 0 || stable_plateau) {
                    ttf_mins = 0.0;
                    status = "NORMAL (Stable/Plateau)";
                    cout << setw(11) << "0.0 mins" << " | " << status;
                } else {
                    ttf_mins = (FAILURE_THRESHOLD - pt.temp) / m;
                    status = "WARNING";
                    cout << setw(11) << ttf_mins << " mins | " << status;
                    
                    double projected_temp = pt.temp + (m * 30.0);
                    if (projected_temp > FAILURE_THRESHOLD) {
                        cout << " -> CRITICAL: Breach in < 30m!";
                    }
                }
                cout << "\n";

                // --- 3. LOG REAL RESULTS TO SQLITE ---
                string sql_insert = "INSERT INTO predictions (time_sec, temperature, state, ttf_mins, status) VALUES (" +
                                    to_string(pt.time_sec) + ", " +
                                    to_string(pt.temp) + ", " +
                                    to_string(state) + ", " +
                                    to_string(ttf_mins) + ", '" +
                                    status + "');";

                sqlite3_exec(db, sql_insert.c_str(), 0, 0, &errMessage);
            }
        }
    }

    file.close();
    sqlite3_close(db);
    cout << "\nSimulation complete. Live terminal output matched and saved to SQLite!\n";
    return 0;
}
