#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace std;

// Define complex number type
using Complex = complex<double>;

// --- Simulation Parameters (Units with hbar=1, m=1) ---
const double HBAR = 1.0;
const double MASS = 1.0;
const double L = 20.0;     // Spatial domain size [-L/2, L/2]
const int NX = 1200;        // Number of spatial grid points
const double DX = L / (NX - 1); // Spatial step size
const double E_trial = 12.5; // Trial energy value
const double TIME_T = 3.141/12.5; // Total time interval for one iteration
const double DT = 0.01;   // Time step size
const int NT = static_cast<int>(TIME_T / DT); // Number of time steps

const int N_ITERATIONS = 400; // Number of iterations for convergence
 
double potential(double x) {
    // Quartic potential with multiplier 0.25   
    return 0.25 * x * x * x * x - E_trial;
}

// Global vectors for the Hamiltonian components (real values)
vector<double> H_diag(NX);
vector<double> H_offdiag(NX - 1); // Assumes boundary conditions (e.g., Dirichlet)

/**
 * @brief Initializes the 1D Hamiltonian matrix components for a general potential V(x).
 * * The Hamiltonian is H = -hbar^2/(2m) * d^2/dx^2 + V(x).
 * Using finite difference, the matrix H is tridiagonal:
 * H_j,j = (hbar^2 / (m * dx^2)) + V(x_j)
 * H_j,j+1 = H_j,j-1 = -hbar^2 / (2m * dx^2)
 */
void initialize_hamiltonian() {
    // Constant factor for the kinetic energy term
    const double KINETIC_FACTOR = HBAR * HBAR / (2.0 * MASS * DX * DX);

    for (int j = 0; j < NX; ++j) {
        double x = j * DX - L / 2.0; // Map index j to physical position x
        
        // Diagonal element: H_j,j
        H_diag[j] = 2.0 * KINETIC_FACTOR + potential(x);

        // Off-diagonal element (H_j,j+1 = H_j+1,j):
        if (j < NX - 1) {
            H_offdiag[j] = -KINETIC_FACTOR;
        }
    }
    
    // Apply boundary conditions (Dirichlet: psi(0)=0, psi(L)=0).
    // The TDMA solver handles the matrix structure for these boundary conditions.
    // Since H_diag[0] and H_diag[NX-1] correspond to boundary points,
    // the system effectively solves for the NX-2 interior points.
    // The current setup includes the boundary points, which is fine for the
    // tridiagonal system as long as the matrix setup is consistent.
}

/**
 * @brief Complex Tridiagonal Matrix Algorithm (TDMA / Thomas Algorithm).
 * Solves A * x = d, where A is a tridiagonal matrix defined by sub, diag, and sup.
 * * @param sub Sub-diagonal (a_j, NX-1 elements).
 * @param diag Main diagonal (b_j, NX elements).
 * @param sup Super-diagonal (c_j, NX-1 elements).
 * @param d Right-hand side vector (d_j, NX elements).
 * @param x Solution vector (x_j, NX elements).
 */
void solve_tridiagonal(const vector<Complex>& sub, const vector<Complex>& diag,
                       const vector<Complex>& sup, const vector<Complex>& d,
                       vector<Complex>& x) {
    if (NX <= 0) return;

    // Modified diagonals
    vector<Complex> c_prime(NX - 1);
    vector<Complex> d_prime(NX);

    // Forward Elimination
    // First equation: b_0 * x_0 + c_0 * x_1 = d_0
    if (diag[0] == Complex(0.0)) {
        cerr << "Error: Diagonal element is zero at index 0." << endl;
        return;
    }
    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = d[0] / diag[0];

    // Main elimination loop
    for (int i = 1; i < NX; ++i) {
        Complex m = diag[i] - sub[i - 1] * c_prime[i - 1];
        if (m == Complex(0.0)) {
            cerr << "Error: Division by zero during TDMA at index " << i << endl;
            return;
        }
        d_prime[i] = (d[i] - sub[i - 1] * d_prime[i - 1]) / m;
        
        if (i < NX - 1) {
            c_prime[i] = sup[i] / m;
        }
    }

    // Back Substitution
    // Last element
    x[NX - 1] = d_prime[NX - 1];

    // Main substitution loop
    for (int i = NX - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}


/**
 * @brief Runs the time-averaging iteration.
 * * The core of the Crank-Nicolson algorithm:
 * (I + i*C*H) * psi_k+1 = (I - i*C*H) * psi_k
 * A * psi_k+1 = R
 * * @param psi_start The initial wave function for this iteration.
 * @return The normalized, time-averaged wave function psi_avg.
 */
vector<Complex> run_time_averaging_iteration(const vector<Complex>& psi_start) {
    vector<Complex> psi_k = psi_start;
    vector<Complex> psi_sum(NX, 0.0);
    
    // Constant factor C = dt / (2 * hbar)
    const Complex C = Complex(0.0, DT / (2.0 * HBAR));

    // Components of the A (LHS) matrix:
    // A = I + i*C*H -> A = sub_A, diag_A, sup_A
    vector<Complex> diag_A(NX);
    vector<Complex> sub_A(NX - 1);
    vector<Complex> sup_A(NX - 1);

    // Components of the B (RHS operator) matrix:
    // B = I - i*C*H -> B = sub_B, diag_B, sup_B
    // B is not explicitly built; we calculate R = B * psi_k directly.
    
    // 1. Build the LHS matrix A components (constant throughout propagation)
    for (int j = 0; j < NX; ++j) {
        diag_A[j] = 1.0 + C * H_diag[j];
        if (j < NX - 1) {
            sub_A[j] = C * H_offdiag[j]; // Off-diagonals are the same for sub and sup
            sup_A[j] = C * H_offdiag[j];
        }
    }

    // 2. Time Propagation Loop (NT steps)
    for (int k = 0; k < NT; ++k) {
        // --- Step 2a: Calculate RHS vector R = B * psi_k ---
        vector<Complex> R(NX);
        for (int j = 0; j < NX; ++j) {
            // Diagonal component (I - i*C*H)_j,j * psi_k_j
            Complex R_diag_term = (1.0 - C * H_diag[j]) * psi_k[j];
            R[j] = R_diag_term;

            // Off-diagonal components (I - i*C*H)_j,j+-1 * psi_k_j+-1
            // Lower (j-1, j)
            if (j > 0) {
                R[j] += (-C * H_offdiag[j - 1]) * psi_k[j - 1];
            }
            // Upper (j+1, j)
            if (j < NX - 1) {
                R[j] += (-C * H_offdiag[j]) * psi_k[j + 1];
            }
        }
        
        // --- Step 2b: Solve the system A * psi_k+1 = R ---
        vector<Complex> psi_k_plus_1(NX);
        solve_tridiagonal(sub_A, diag_A, sup_A, R, psi_k_plus_1);

        // --- Step 2c: Accumulate for averaging ---
        for (int j = 0; j < NX; ++j) {
            psi_sum[j] += psi_k_plus_1[j];
        }

        // Prepare for the next step
        psi_k = psi_k_plus_1;
    }

    // --- Step 3: Calculate the Time Average ---
    vector<Complex> psi_avg(NX);
    for (int j = 0; j < NX; ++j) {
        // Average is (1/NT) * sum(psi_j)
        psi_avg[j] = psi_sum[j] / (double)NT;
    }

    return psi_avg;
}

/**
 * @brief Normalizes a wave function (L2 norm).
 * @param psi The wave function vector to normalize.
 */
void normalize_psi(vector<Complex>& psi) {
    double norm_sq = 0.0;
    for (const auto& val : psi) {
        norm_sq += norm(val); // norm(z) = z * conj(z) = real(z)^2 + imag(z)^2
    }
    
    // Add spatial integration factor dx to the norm squared
    norm_sq *= DX;
    double norm = sqrt(norm_sq);

    if (norm < 1e-12) {
        cerr << "Warning: Wave function norm is close to zero. Cannot normalize." << endl;
        return;
    }

    // Normalize: psi' = psi / norm
    for (auto& val : psi) {
        val /= norm;
    }
}

/**
 * @brief Calculates the expectation value of the Hamiltonian <H> for a given psi.
 * @param psi The normalized wave function.
 * @return The expectation value of the energy (real part).
 */
double calculate_energy(const vector<Complex>& psi) {
    Complex energy_complex = 0.0;

    // <H> = sum_j (psi_j*) * (H * psi)_j * dx
    for (int j = 0; j < NX; ++j) {
        // Calculate (H * psi)_j
        Complex H_psi_j = H_diag[j] * psi[j];
        if (j > 0) {
            H_psi_j += H_offdiag[j - 1] * psi[j - 1]; // Lower
        }
        if (j < NX - 1) {
            H_psi_j += H_offdiag[j] * psi[j + 1]; // Upper
        }
        
        // Sum (psi_j*) * (H * psi)_j
        energy_complex += conj(psi[j]) * H_psi_j;
    }
    
    // Final result includes DX factor
    return (energy_complex * DX).real();
}

/**
 * @brief Calculates the L2 difference between two wave functions.
 */
double calculate_difference(const vector<Complex>& psi_new, const vector<Complex>& psi_old) {
    double diff_sq = 0.0;
    for (int j = 0; j < NX; ++j) {
        Complex diff = psi_new[j] - psi_old[j];
        diff_sq += norm(diff);
    }
    return sqrt(diff_sq * DX);
}


// --- Main Function ---
int main() {
    cout << "--- 1D Quantum Time Propagation and Averaging for Eigenstate of H" << endl;
    cout << "Domain: " << -L/2.0 << " to " << L/2.0 << " (NX=" << NX << ", dx=" << DX << ")" << endl;
    cout << "Propagation Time T: " << TIME_T << " (NT=" << NT << ", dt=" << DT << ")" << endl;
    cout << "Iterations: " << N_ITERATIONS << endl;
    
    // 1. Initialize Hamiltonian
    initialize_hamiltonian();

    // 2. Initialize the wave function (e.g., Gaussian centered at x=0)
    vector<Complex> psi(NX);
    double initial_width = L / 10.0;
    for (int j = 0; j < NX; ++j) {
        double x = j * DX - L / 2.0;
        // Initial state is a product of Gaussian and sine function
        double gaussian = sin(x-0.5)*exp(-(x-0.5) * (x-0.5) / (2.0 * initial_width * initial_width));
        // Add a random phase (optional, but good for mixing states)
        psi[j] = Complex(gaussian, 0.0);
    }
    normalize_psi(psi);

    cout << fixed << setprecision(8);
    cout << "\nIteration | Energy <H> | L2 Diff (from prev) | Avg | Max(|psi|^2)" << endl;
    cout << "----------|------------|---------------------|-----|-------------" << endl;

    // 3. Iteration Loop
    for (int i = 0; i < N_ITERATIONS; ++i) {
        vector<Complex> psi_old = psi;

        // Perform time propagation, averaging, and get the new average
        vector<Complex> psi_avg = run_time_averaging_iteration(psi_old);

        // Normalize the averaged wave function
        normalize_psi(psi_avg);
        
        // Update psi for the next iteration
        psi = psi_avg;

        // Calculate metrics
        double energy = calculate_energy(psi);
        double diff = calculate_difference(psi, psi_old);
        
        // Output convergence info
        double max_prob = 0.0;
        for (const auto& val : psi) {
            max_prob = max(max_prob, norm(val));
        }

        cout << setw(9) << i << " | "
             << setw(10) << energy + E_trial << " | "
             << setw(19) << diff << " | "
             << setw(3) << " " << "| "
             << setw(11) << max_prob << endl;
    }

    // 4. Output Final State Data (for plotting/verification)
    cout << "\n--- Final Converged Wave Function (|psi|^2) Data ---" << endl;
    cout << "x, probability_density" << endl;
    for (int j = 0; j < NX; ++j) {
        double x = j * DX - L / 2.0;
        double prob_density = norm(psi[j]);
        cout << x << ", " << prob_density << endl;
    }

    cout << "\nProgram finished." << endl;
    
    return 0;
}
