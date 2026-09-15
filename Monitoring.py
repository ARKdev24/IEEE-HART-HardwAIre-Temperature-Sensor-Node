#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace std;

struct DataPoint {
    double time_sec;
    double temp;
};

// --- Mathematical Prediction Models ---

// Linear Regression: y = mx + c. Returns slope (m).
double getLinearSlope(const deque<DataPoint>& window) {
    if (window.size() < 2) return 0.0;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = window.size();
    double t0 = window.front().time_sec; 

    for (const auto& pt : window) {
        double x = pt.time_sec - t0;
        sum_x += x;
        sum_y += pt.temp;
        sum_xy += x * pt.temp;
        sum_xx += x * x;
    }
    double denom = (n * sum_xx) - (sum_x * sum_x);
    if (denom == 0) return 0.0;
    return ((n * sum_xy) - (sum_x * sum_y)) / denom;
}

// Exponential Regression: y = a * e^(bx) -> ln(y) = ln(a) + bx. Returns exponent (b).
double getExponentialRate(const deque<DataPoint>& window) {
    if (window.size() < 2) return 0.0;
    double sum_x = 0, sum_lny = 0, sum_xlny = 0, sum_xx = 0;
    int n = window.size();
    double t0 = window.front().time_sec; 

    for (const auto& pt : window) {
        double x = pt.time_sec - t0;
        double lny = log(pt.temp);
        sum_x += x;
        sum_lny += lny;
        sum_xlny += x * lny;
        sum_xx += x * x;
    }
    double denom = (n * sum_xx) - (sum_x * sum_x);
    if (denom == 0) return 0.0;
    return ((n * sum_xlny) - (sum_x * sum_lny)) / denom;
}

// --- 2D ASCII Graph Plotter ---
void drawGraphAndStats(const deque<DataPoint>& window, double current_time, int current_state) {
    const int WIDTH = 80;
    const int HEIGHT = 15;
    const double MIN_T = 2.5, MAX_T = 8.5;
    const double WINDOW_SEC = 3 * 3600.0; // 3 hours

    char grid[HEIGHT][WIDTH];
    for (int r = 0; r < HEIGHT; r++) 
        for (int c = 0; c < WIDTH; c++) 
            grid[r][c] = ' ';

    double window_start = current_time - WINDOW_SEC;

    // Map data points to grid
    for (const auto& pt : window) {
        if (pt.time_sec < window_start) continue;

        int col = (int)(((pt.time_sec - window_start) / WINDOW_SEC) * (WIDTH - 1));
        int row = HEIGHT - 1 - (int)(((pt.temp - MIN_T) / (MAX_T - MIN_T)) * (HEIGHT - 1));
        
        // Clamp to prevent out-of-bounds
        col = max(0, min(WIDTH - 1, col));
        row = max(0, min(HEIGHT - 1, row));
        
        grid[row][col] = '*';
    }

    // Clear terminal (works on Windows/Linux/Mac)
    cout << "\033[2J\033[1;1H"; 
    cout << "=== 3-Hour Sliding Window Temperature Monitor ===\n";

    // Draw Grid with Y-axis labels
    for (int r = 0; r < HEIGHT; r++) {
        double temp_label = MAX_T - (r * (MAX_T - MIN_T) / (HEIGHT - 1));
        cout << fixed << setprecision(1) << temp_label << "C |";
        for (int c = 0; c < WIDTH; c++) {
            // Draw warning threshold line at 7.5 and 8.0
            if (grid[r][c] == ' ') {
                if (abs(temp_label - 8.0) < 0.1) cout << "-";
                else if (abs(temp_label - 7.5) < 0.1) cout << ".";
                else cout << " ";
            } else {
                cout << grid[r][c];
            }
        }
        cout << "\n";
    }
    
    // X-axis
    cout << "      +";
    for (int c = 0; c < WIDTH; c++) cout << "-";
    cout << "\n      -3 Hours                                     Current Time\n\n";

    // --- Output Stats & Predictions ---
    double cur_t = window.back().temp;
    cout << "Current State: " << current_state << " | Temp: " << fixed << setprecision(3) << cur_t << " C\n";
    
    if (current_state > 0) {
        double m = getLinearSlope(window);
        double b = getExponentialRate(window);

        cout << ">> WARNING ZONE DETECTED <<\n";
        
        if (m > 0) {
            double linear_ttf = (8.0 - cur_t) / m;
            cout << "Linear Prediction TTF     : " << (linear_ttf / 60.0) << " minutes\n";
        } else {
            cout << "Linear Prediction         : System is stabilizing/cooling.\n";
        }

        if (b > 0) {
            // Calculate exponential TTF: 8.0 = current_temp * e^(b * time_left)
            double exp_ttf = log(8.0 / cur_t) / b;
            cout << "Exponential Prediction TTF: " << (exp_ttf / 60.0) << " minutes\n";
        }
    } else {
        cout << "Status: Normal Operation.\n";
    }
}

int main() {
    ifstream file("sensor_data.csv");
    if (!file.is_open()) {
        cout << "Error: Run dataset generator first to create sensor_data.csv\n";
        return 1;
    }

    string line;
    getline(file, line); // Skip header

    deque<DataPoint> window;
    const double WINDOW_SEC = 3 * 3600;

    // Fast-forward variables to skip boring normal data and get straight to Day 4 action
    const double FAST_FORWARD_UNTIL = (3 * 24 * 3600) + (4 * 3600); 

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

        // Slide the window
        while (!window.empty() && (pt.time_sec - window.front().time_sec > WINDOW_SEC)) {
            window.pop_front();
        }

        // Fast-forward past the boring first 3 days to show you the interesting part
        if (pt.time_sec < FAST_FORWARD_UNTIL) continue;

        // Draw graph and pause so you can see it animate
        drawGraphAndStats(window, pt.time_sec, state);
        
        // Speed up the animation based on sampling rate 
        // (Warning zones update faster on screen to match real life)
        this_thread::sleep_for(chrono::milliseconds(sample_rate / 2)); 
    }

    file.close();
    return 0;
}
