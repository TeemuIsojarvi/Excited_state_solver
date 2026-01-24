#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <complex>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <functional>

using namespace std;

using Complex = complex<double>;

// --- Global Simulation Constants ---
const double HBAR = 1.0;
const double MASS = 1.0;
const double L = 3.4;     // Spatial domain size [-L/2, L/2] in both dimensions
const double TIME_T = 3.141/11.0; // Total time interval for one iteration
double DT = 0.02;   // Time step size
const int NT = static_cast<int>(TIME_T / DT); // Number of time steps
const int N_ITERATIONS = 600; // Total number of outer iterations for convergence
constexpr std::array<int, 3> Refine_points = {525, 560, 580}; // Iteration numbers where discretization is refined

// Initial grid points (Coarse grid)
const int NX_COARSE = 120;
const int NY_COARSE = 120;

const double E_trial = 11.0; // Attempting to converge to an eigenstate of H with energy closest to this value

double potential(double x, double y) {
    // Particle in a supercircular area with finite potential step at boundaries
    if(pow(abs(x),3) + pow(abs(y),3) < pow(1.063999,3)) return -E_trial;
    else return 20000.0;
}

// Structure to hold grid parameters and derived factors dynamically
struct GridParams {
    int NX;
    int NY;
    int NTOTAL;
    double DX;
    double DY;
    double KINETIC_FACTOR_X;
    double KINETIC_FACTOR_Y;
};

// --- Function Prototypes ---
void solve_tridiagonal(const vector<Complex>& sub, const vector<Complex>& diag,
                       const vector<Complex>& sup, const vector<Complex>& d,
                       vector<Complex>& x, int N);

void normalize_psi(vector<Complex>& psi, const GridParams& params);

vector<Complex> run_adi_iteration(const vector<Complex>& psi_start, const GridParams& params);

double calculate_energy(const vector<Complex>& psi, const GridParams& params);

double calculate_difference(const vector<Complex>& psi_new, const vector<Complex>& psi_old, const GridParams& params);

void save_data_to_files(const vector<Complex>& psi, const GridParams& params);

Complex cubic_hermite_interpolation_1d(double x_target, double x_start, double dx, const vector<Complex>& psi_1d, int N);

vector<Complex> interpolate_2d_bicubic(const vector<Complex>& psi_old, const GridParams& old_params, const GridParams& new_params);

// --- Function Definitions ---

/**
 * @brief Complex Tridiagonal Matrix Algorithm (TDMA / Thomas Algorithm). (Unchanged)
 */
void solve_tridiagonal(const vector<Complex>& sub, const vector<Complex>& diag,
                       const vector<Complex>& sup, const vector<Complex>& d,
                       vector<Complex>& x, int N) {
    if (N <= 0) return;
    vector<Complex> c_prime(N - 1);
    vector<Complex> d_prime(N);
    if (diag[0] == Complex(0.0)) {
        cerr << "TDMA Error: Diagonal element is zero at index 0." << endl;
        return;
    }
    if (N == 1) {
        x[0] = d[0] / diag[0];
        return;
    }
    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = d[0] / diag[0];
    for (int i = 1; i < N; ++i) {
        Complex m = diag[i] - sub[i - 1] * c_prime[i - 1];
        if (m == Complex(0.0)) {
            cerr << "TDMA Error: Division by zero during TDMA at index " << i << endl;
            return;
        }
        d_prime[i] = (d[i] - sub[i - 1] * d_prime[i - 1]) / m;
        if (i < N - 1) {
            c_prime[i] = sup[i] / m;
        }
    }
    x[N - 1] = d_prime[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}

/**
 * @brief Normalizes a wave function (L2 norm). (Modified to use GridParams)
 */
void normalize_psi(vector<Complex>& psi, const GridParams& params) {
    double norm_sq = 0.0;
    for (const auto& val : psi) {
        norm_sq += norm(val);
    }
    norm_sq *= (params.DX * params.DY); // dV = dx * dy
    double norm_val = sqrt(norm_sq);

    if (norm_val < 1e-12) {
        cerr << "Warning: Wave function norm is close to zero. Cannot normalize." << endl;
        fill(psi.begin(), psi.end(), Complex(0.0));
        return;
    }

    for (auto& val : psi) {
        val /= norm_val;
    }
}

/**
 * @brief Runs the time-averaging iteration using the ADI scheme. (Modified to use GridParams)
 */
vector<Complex> run_adi_iteration(const vector<Complex>& psi_start, const GridParams& params) {
    vector<Complex> psi_k = psi_start;
    vector<Complex> psi_sum(params.NTOTAL, 0.0);
    
    const Complex C = Complex(0.0, DT / (2.0 * HBAR));

    vector<Complex> diag(max(params.NX, params.NY));
    vector<Complex> sub(max(params.NX, params.NY) - 1);
    vector<Complex> sup(max(params.NX, params.NY) - 1);
    vector<Complex> R(max(params.NX, params.NY));
    vector<Complex> solution(max(params.NX, params.NY));

    const double tx_diag_val = 2.0 * params.KINETIC_FACTOR_X;
    const double tx_offdiag_val = -params.KINETIC_FACTOR_X;
    const double ty_diag_val = 2.0 * params.KINETIC_FACTOR_Y;
    const double ty_offdiag_val = -params.KINETIC_FACTOR_Y;

    for (int k = 0; k < NT; ++k) {
        vector<Complex> psi_half(params.NTOTAL);

        // 1. X-SWEEP (Implicit in X, Explicit in Y)
        for (int jy = 0; jy < params.NY; ++jy) {
            double y = jy * params.DY - L / 2.0;
            
            // 1a. Calculate RHS R_x = [I - i*C*H_y] psi_k
            for (int jx = 0; jx < params.NX; ++jx) {
                int j = jx + jy * params.NX;
                double x = jx * params.DX - L / 2.0;
                
                double V_half = 0.5 * potential(x, y);
                Complex R_diag_term = (1.0 - C * (ty_diag_val + V_half)) * psi_k[j];
                R[jx] = R_diag_term;

                if (jy > 0) {
                    R[jx] += (-C * ty_offdiag_val) * psi_k[j - params.NX];
                }
                if (jy < params.NY - 1) {
                    R[jx] += (-C * ty_offdiag_val) * psi_k[j + params.NX];
                }
            }

            // 1b. Build LHS A_x = [I + i*C*H_x] and solve
            for (int jx = 0; jx < params.NX; ++jx) {
                double x = jx * params.DX - L / 2.0;
                double V_half = 0.5 * potential(x, y);

                diag[jx] = 1.0 + C * (tx_diag_val + V_half);
                
                if (jx < params.NX - 1) {
                    sub[jx] = C * tx_offdiag_val;
                    sup[jx] = C * tx_offdiag_val;
                }
            }
            
            if (jy == 0 || jy == params.NY - 1) {
                 fill(solution.begin(), solution.begin() + params.NX, Complex(0.0));
            } else {
                 solve_tridiagonal(sub, diag, sup, R, solution, params.NX);
            }
            
            for (int jx = 0; jx < params.NX; ++jx) {
                psi_half[jx + jy * params.NX] = solution[jx];
            }
        }


        // 2. Y-SWEEP (Implicit in Y, Explicit in X)
        for (int jx = 0; jx < params.NX; ++jx) {
            double x = jx * params.DX - L / 2.0;

            // 2a. Calculate RHS R_y = [I - i*C*H_x] psi_k+1/2
            for (int jy = 0; jy < params.NY; ++jy) {
                int j = jx + jy * params.NX;
                double y = jy * params.DY - L / 2.0;

                double V_half = 0.5 * potential(x, y);

                Complex R_diag_term = (1.0 - C * (tx_diag_val + V_half)) * psi_half[j];
                R[jy] = R_diag_term;

                if (jx > 0) {
                    R[jy] += (-C * tx_offdiag_val) * psi_half[j - 1];
                }
                if (jx < params.NX - 1) {
                    R[jy] += (-C * tx_offdiag_val) * psi_half[j + 1];
                }
            }

            // 2b. Build LHS A_y = [I + i*C*H_y] and solve
            for (int jy = 0; jy < params.NY; ++jy) {
                double y = jy * params.DY - L / 2.0;
                double V_half = 0.5 * potential(x, y);

                diag[jy] = 1.0 + C * (ty_diag_val + V_half);

                if (jy < params.NY - 1) {
                    sub[jy] = C * ty_offdiag_val;
                    sup[jy] = C * ty_offdiag_val;
                }
            }
            
            if (jx == 0 || jx == params.NX - 1) {
                fill(solution.begin(), solution.begin() + params.NY, Complex(0.0));
            } else {
                solve_tridiagonal(sub, diag, sup, R, solution, params.NY);
            }

            for (int jy = 0; jy < params.NY; ++jy) {
                int j = jx + jy * params.NX;
                psi_k[j] = solution[jy];
            }
        }
        
        // 2c: Accumulate for averaging
        for (int j = 0; j < params.NTOTAL; ++j) {
            psi_sum[j] += psi_k[j];
        }
    }

    // 3: Calculate the Time Average
    vector<Complex> psi_avg(params.NTOTAL);
    for (int j = 0; j < params.NTOTAL; ++j) {
        psi_avg[j] = psi_sum[j] / (double)NT;
    }

    return psi_avg;
}

/**
 * @brief Calculates the expectation value of the Hamiltonian <H>. (Modified to use GridParams)
 */
double calculate_energy(const vector<Complex>& psi, const GridParams& params) {
    Complex energy_complex = 0.0;

    for (int jy = 0; jy < params.NY; ++jy) {
        for (int jx = 0; jx < params.NX; ++jx) {
            int j = jx + jy * params.NX;
            double x = jx * params.DX - L / 2.0;
            double y = jy * params.DY - L / 2.0;
            
            Complex H_psi_j = potential(x, y) * psi[j];

            // T_x term
            Complex T_x_psi = Complex(0.0);
            T_x_psi += (2.0 * params.KINETIC_FACTOR_X) * psi[j];
            if (jx > 0)   T_x_psi += (-params.KINETIC_FACTOR_X) * psi[j - 1];
            if (jx < params.NX - 1) T_x_psi += (-params.KINETIC_FACTOR_X) * psi[j + 1];

            // T_y term
            Complex T_y_psi = Complex(0.0);
            T_y_psi += (2.0 * params.KINETIC_FACTOR_Y) * psi[j];
            if (jy > 0)   T_y_psi += (-params.KINETIC_FACTOR_Y) * psi[j - params.NX];
            if (jy < params.NY - 1) T_y_psi += (-params.KINETIC_FACTOR_Y) * psi[j + params.NX];

            H_psi_j += T_x_psi;
            H_psi_j += T_y_psi;

            energy_complex += conj(psi[j]) * H_psi_j;
        }
    }
    
    return (energy_complex * params.DX * params.DY).real();
}

/**
 * @brief Calculates the L2 difference between two wave functions. (Modified to use GridParams)
 */
double calculate_difference(const vector<Complex>& psi_new, const vector<Complex>& psi_old, const GridParams& params) {
    double diff_sq = 0.0;
    // Note: The L2 difference is calculated on the same grid (before interpolation)
    for (int j = 0; j < params.NTOTAL; ++j) {
        Complex diff = psi_new[j] - psi_old[j];
        diff_sq += norm(diff);
    }
    return sqrt(diff_sq * params.DX * params.DY);
}


/**
 * @brief Saves the final probability density and the potential grid to separate CSV files. (Modified to use GridParams)
 */
void save_data_to_files(const vector<Complex>& psi, const GridParams& params) {
    // 1. Save Probability Density (|psi|^2)
    ofstream psi_file("psi_2d_data.csv");
    if (!psi_file.is_open()) {
        cerr << "Error: Could not open psi_2d_data.csv for writing." << endl;
        return;
    }

    psi_file << "x_coord,y_coord,probability_density" << endl;
    for (int jy = 0; jy < params.NY; ++jy) {
        double y = jy * params.DY - L / 2.0;
        for (int jx = 0; jx < params.NX; ++jx) {
            double x = jx * params.DX - L / 2.0;
            int j = jx + jy * params.NX;
            double prob_density = norm(psi[j]);
            psi_file << fixed << setprecision(6) << x << "," << y << "," << prob_density << endl;
        }
    }
    psi_file.close();
    cout << "\nData saved to psi_2d_data.csv (NX=" << params.NX << ")" << endl;

    // 2. Save Potential Energy V(x,y)
    ofstream V_file("potential_2d_data.csv");
    if (!V_file.is_open()) {
        cerr << "Error: Could not open potential_2d_data.csv for writing." << endl;
        return;
    }

    V_file << "x_coord,y_coord,potential_energy" << endl;
    for (int jy = 0; jy < params.NY; ++jy) {
        double y = jy * params.DY - L / 2.0;
        for (int jx = 0; jx < params.NX; ++jx) {
            double x = jx * params.DX - L / 2.0;
            double V_val = potential(x, y);
            V_file << fixed << setprecision(6) << x << "," << y << "," << V_val << endl;
        }
    }
    V_file.close();
    cout << "Data saved to potential_2d_data.csv" << endl;
}

/**
 * @brief 1D Cubic Hermite Spline Interpolation for Complex Numbers.
 * Requires four adjacent points: psi[i-1], psi[i], psi[i+1], psi[i+2].
 * Uses centered finite differences to estimate derivatives (slopes).
 * @param x_target The coordinate on the 1D grid to interpolate to.
 * @param x_start The coordinate of the first point (index 0).
 * @param dx The spacing between grid points.
 * @param psi_1d The 1D vector of Complex grid values.
 * @param N The size of psi_1d.
 * @return The interpolated Complex value.
 */
Complex cubic_hermite_interpolation_1d(double x_target, double x_start, double dx, const vector<Complex>& psi_1d, int N) {
    // 1. Find the interval [i, i+1] where the target point falls
    double u = (x_target - x_start) / dx;
    int i = floor(u);
    
    // Check boundaries (handle extrapolation by clamping to nearest boundary point)
    if (i < 1 || i >= N - 2) {
        // If outside the reliable range for cubic interpolation (requires i-1 to i+2), 
        // fallback to nearest neighbor or linear extrapolation (here, using nearest point)
        if (i < 0) i = 0;
        if (i >= N - 1) i = N - 2; // clamp i to the second-to-last index
        // Linear interpolation for edges (safer than complex clamping)
        if (i == N - 2) i--;
        
        Complex p0 = psi_1d[i];
        Complex p1 = psi_1d[i + 1];
        double t = u - i;
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        return p0 * (1.0 - t) + p1 * t; // Bilinear near edges
    }

    // Normalized coordinate t in [0, 1] within the interval [i, i+1]
    double t = u - i;
    
    // Four control points required for the Hermite spline, indices [i-1, i, i+1, i+2]
    Complex p_m1 = psi_1d[i - 1]; // p_i-1
    Complex p0   = psi_1d[i];     // p_i
    Complex p1   = psi_1d[i + 1];   // p_i+1
    Complex p2   = psi_1d[i + 2]; // p_i+2

    // Estimate derivatives (slopes) at p0 and p1 using Centered Finite Difference
    // m0 = slope at p0 (index i)
    // m1 = slope at p1 (index i+1)
    // Since dx=1 in normalized coordinates, the formula is (p_next - p_prev) / 2
    Complex m0 = (p1 - p_m1) / 2.0; 
    Complex m1 = (p2 - p0) / 2.0;
    
    // Cubic Hermite basis functions
    double t2 = t * t;
    double t3 = t2 * t;
    
    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;
    
    // Interpolated value:
    // P(t) = h00*p0 + h10*m0 + h01*p1 + h11*m1
    return p0 * h00 + m0 * h10 * dx + p1 * h01 + m1 * h11 * dx; 
}


/**
 * @brief 2D Bicubic Interpolation from an old grid to a new grid.
 * Uses 1D cubic Hermite interpolation sequentially (first X-direction, then Y-direction).
 * @param psi_old The wave function on the coarse grid.
 * @param old_params Grid parameters for the coarse grid.
 * @param new_params Grid parameters for the fine grid.
 * @return The interpolated wave function on the fine grid.
 */
vector<Complex> interpolate_2d_bicubic(const vector<Complex>& psi_old, const GridParams& old_params, const GridParams& new_params) {
    cout << "Interpolating from " << old_params.NX << "x" << old_params.NY 
         << " to " << new_params.NX << "x" << new_params.NY << " using Bicubic Interpolation..." << endl;

    vector<Complex> psi_new(new_params.NTOTAL);
    
    // 1. Pre-calculate the values interpolated in X-direction
    // This intermediate grid (size: NX_new x NY_old) holds Complex values
    vector<vector<Complex>> psi_x_interp(old_params.NY, vector<Complex>(new_params.NX));

    double x_start_old = -L / 2.0;

    // Iterate over each OLD row (fixed y_j)
    for (int jy_old = 0; jy_old < old_params.NY; ++jy_old) {
        // Extract the 1D slice of psi_old for this y-row
        vector<Complex> psi_row_old(old_params.NX);
        for(int jx = 0; jx < old_params.NX; ++jx) {
            psi_row_old[jx] = psi_old[jx + jy_old * old_params.NX];
        }

        // Interpolate this row onto the new X-coordinates
        for (int jx_new = 0; jx_new < new_params.NX; ++jx_new) {
            double x_target = jx_new * new_params.DX + x_start_old;
            psi_x_interp[jy_old][jx_new] = cubic_hermite_interpolation_1d(
                x_target, x_start_old, old_params.DX, psi_row_old, old_params.NX
            );
        }
    }

    // 2. Interpolate in Y-direction
    double y_start_old = -L / 2.0;
    
    // Iterate over each NEW column (fixed x_i)
    for (int jx_new = 0; jx_new < new_params.NX; ++jx_new) {
        
        // Extract the 1D slice (column) of the intermediate grid
        vector<Complex> psi_col_interp_x(old_params.NY);
        for (int jy = 0; jy < old_params.NY; ++jy) {
            psi_col_interp_x[jy] = psi_x_interp[jy][jx_new];
        }

        // Interpolate this column onto the new Y-coordinates
        for (int jy_new = 0; jy_new < new_params.NY; ++jy_new) {
            double y_target = jy_new * new_params.DY + y_start_old;
            
            Complex interpolated_value = cubic_hermite_interpolation_1d(
                y_target, y_start_old, old_params.DY, psi_col_interp_x, old_params.NY
            );
            
            // Store result in the new flattened vector
            psi_new[jx_new + jy_new * new_params.NX] = interpolated_value;
        }
    }
    
    return psi_new;
}


// --- Main Function (Refactored) ---
int main() {
    // --- Grid Initialization (Coarse Grid) ---
    GridParams params;
    params.NX = NX_COARSE;
    params.NY = NY_COARSE;
    params.NTOTAL = params.NX * params.NY;
    params.DX = L / (params.NX - 1);
    params.DY = L / (params.NY - 1);
    params.KINETIC_FACTOR_X = HBAR * HBAR / (2.0 * MASS * params.DX * params.DX);
    params.KINETIC_FACTOR_Y = HBAR * HBAR / (2.0 * MASS * params.DY * params.DY);
    
    bool refined = false;

    cout << "--- 2D Quantum Time Propagation and Averaging for Eigenstate Calculation (ADI) ---" << endl;
    cout << "Total Iterations: " << N_ITERATIONS << endl;
    cout << "Initial Grid: " << params.NX << "x" << params.NY << " (dx=" << params.DX << ")" << endl;
    
    // 1. Initialize the wave function (sum of 2D Gaussians)
    vector<Complex> psi(params.NTOTAL);
    double initial_width = L / 3.0;
    for (int jy = 0; jy < params.NY; ++jy) {
        double y = jy * params.DY - L / 2.0;
        for (int jx = 0; jx < params.NX; ++jx) {
            double x = jx * params.DX - L / 2.0;
            int j = jx + jy * params.NX;
            
            double gaussian = exp(-((x-0.2) * (x-0.2) + (y+0.1) * (y+0.1)) / (2.0 * initial_width * initial_width)) - 0.7*exp(-((x+0.45) * (x+0.45) + (y+0.35) * (y+0.35)) / (2.0 * initial_width * initial_width));
            psi[j] = Complex(gaussian, 0.0);
        }
    }
    normalize_psi(psi, params);

    cout << fixed << setprecision(8);
    cout << "\nIteration | Grid Size | Energy <H> | L2 Diff (from prev)" << endl;
    cout << "----------|-----------|------------|---------------------" << endl;

    // 2. Iteration Loop
    for (int i = 0; i < N_ITERATIONS; ++i) {
        // --- ADAPTIVE GRID REFINEMENT STEP ---
        if (std::find(Refine_points.begin(), Refine_points.end(), i) != Refine_points.end()) {
            cout << "\n--- GRID REFINEMENT TRIGGERED  at iteration " << i << " ---" << endl;
            
            // Save old parameters
            GridParams old_params = params;
            
            DT /= 1.5;

            // Calculate new parameters (halving step size means NX_new = 2*NX_old - 1)
            GridParams new_params;
            new_params.NX = 2 * old_params.NX - 1;
            new_params.NY = 2 * old_params.NY - 1;
            new_params.NTOTAL = new_params.NX * new_params.NY;
            new_params.DX = L / (new_params.NX - 1); // Should be exactly half of old DX
            new_params.DY = L / (new_params.NY - 1);
            
            // Calculate new kinetic factors
            new_params.KINETIC_FACTOR_X = HBAR * HBAR / (2.0 * MASS * new_params.DX * new_params.DX);
            new_params.KINETIC_FACTOR_Y = HBAR * HBAR / (2.0 * MASS * new_params.DY * new_params.DY);

            // Interpolate the wave function
            vector<Complex> psi_new = interpolate_2d_bicubic(psi, old_params, new_params);
            
            // Update the main wave function and grid parameters
            psi = psi_new;
            params = new_params;
            normalize_psi(psi, params); // Re-normalize after interpolation
            
            refined = true;
            
            cout << "New Grid: " << params.NX << "x" << params.NY << " (dx=" << params.DX << ")" << endl;
            cout << "--------------------------------------------------------" << endl;
        }
        
        vector<Complex> psi_old = psi;

        // Perform ADI time propagation
        vector<Complex> psi_avg = run_adi_iteration(psi_old, params);

        // Normalize the averaged wave function
        normalize_psi(psi_avg, params);
        
        // Update psi for the next iteration
        psi = psi_avg;

        // Calculate metrics
        double energy = calculate_energy(psi, params);
        
        // Only calculate difference if we didn't just interpolate (as interpolation dramatically increases the diff)
        double diff = (std::find(Refine_points.begin(), Refine_points.end(), i) != Refine_points.end() || i == 0) ? calculate_difference(psi, psi_old, params) : 0.0;

        cout << setw(9) << i << " | "
             << setw(9) << params.NX << "x" << params.NY << " | "
             << setw(10) << energy + E_trial << " | "
             << setw(19) << diff << endl;

    }

    // 3. Save Final State Data
    save_data_to_files(psi, params);

    cout << "\nProgram finished." << endl;
    
    return 0;
}
