import numpy as np
import pandas as pd
import matplotlib

# Use a noninteractive backend so we never stall on GUI windows:
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

#################################################
# 1. Load data
#################################################
df = pd.read_csv("problem_fits_data_reduced.csv")  # Must be in the working directory


#################################################
# 2. Define the 4-parameter logistic
#################################################
def logistic_4p(x, A, B, M, T):
    """
    4-Parameter Logistic:
      f(x) = A + (B - A) / [1 + exp(-M*(x - T))]
    A = lower asymptote
    B = upper asymptote
    M = slope
    T = inflection (horizontal midpoint)
    """
    return A + (B - A) / (1.0 + np.exp(-M * (x - T)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


#################################################
# 3. Heuristic guesses for (A, B, M, T)
#################################################
def guess_4p_params(x_data, y_data):
    """
    Builds a single (A, B, M, T) guess from min/max stats.
    We'll refine it with random restarts in the fit function.
    """
    # Sort data just for consistent references
    idx_sort = np.argsort(x_data)
    x_sorted = x_data[idx_sort]
    y_sorted = y_data[idx_sort]

    x_min, x_max = x_sorted[0], x_sorted[-1]
    y_min, y_max = y_sorted.min(), y_sorted.max()

    # If there's no variation in y, set A ~ B ~ that value
    if np.isclose(y_min, y_max, atol=1e-9):
        A0 = 0.95 * y_min
        B0 = 1.05 * y_min
        M0 = 0.0
        T0 = 0.5 * (x_min + x_max)
        return (A0, B0, M0, T0)

    # A ~ y_min, B ~ y_max
    A0 = y_min
    B0 = y_max

    # For M, pick sign from whether y is increasing overall
    slope_sign = np.sign(y_sorted[-1] - y_sorted[0])
    if slope_sign == 0:
        slope_sign = 1  # fallback

    # Magnitude ~ 4 / range
    x_range = max(1e-9, x_max - x_min)
    M0 = slope_sign * (4.0 / x_range)

    # T ~ x midpoint
    T0 = 0.5 * (x_min + x_max)

    return (A0, B0, M0, T0)


#################################################
# 4. Fit with random restarts, no skipping
#################################################
def fit_4p_logistic_noskip(x_data, y_data, n_starts=5):
    """
    - We'll do multiple random initial guesses around heuristic guesses.
    - We'll do unbounded curve_fit(method='lm'). If it fails for some guess,
      we keep trying. If all fail, we fallback to the heuristic.
    - Returns (best_params, best_rmse) for [A, B, M, T].
    """
    # Base heuristic
    base_guess = guess_4p_params(x_data, y_data)

    best_params = base_guess
    y_pred_base = logistic_4p(x_data, *base_guess)
    best_rmse_val = rmse(y_data, y_pred_base)

    # unbounded => method='lm'
    for _ in range(n_starts):
        # Create a random guess near the base guess
        # We'll allow up to +/- 20% variation from the base guess for each param
        # except slope sign might flip, so let's allow a bigger random range for M
        A0, B0, M0, T0 = base_guess

        # random scaling:
        A_rand = A0 * np.random.uniform(0.8, 1.2)
        B_rand = B0 * np.random.uniform(0.8, 1.2)
        # allow slope to vary more widely
        M_rand = M0 * np.random.uniform(0.5, 2.0)  # factor up to 2
        if np.random.rand() < 0.1:
            M_rand = -M_rand  # sometimes flip sign

        T_rand = T0 + np.random.uniform(-0.2, 0.2) * (x_data.max() - x_data.min())

        guess = [A_rand, B_rand, M_rand, T_rand]

        try:
            popt, _ = curve_fit(
                logistic_4p, x_data, y_data, p0=guess, method="lm", maxfev=2000
            )
            # Evaluate new RMSE
            y_pred_new = logistic_4p(x_data, *popt)
            cur_rmse = rmse(y_data, y_pred_new)

            if cur_rmse < best_rmse_val:
                best_rmse_val = cur_rmse
                best_params = popt
        except (RuntimeError, ValueError):
            # curve_fit can fail, we ignore and move on
            continue

    # Return the best found
    return best_params, best_rmse_val


#################################################
# 5. Process all series: ALWAYS produce results
#################################################
grouped = df.groupby("seriesNumber")
results = []

for sid, grp in grouped:
    x_data = grp["Year"].values
    y_data = grp["Value"].values

    # If only 1 point, we can still do a trivial logistic guess
    # We'll do it anyway, no skip
    best_params, best_rmse_val = fit_4p_logistic_noskip(x_data, y_data, n_starts=10)

    # Compare to linear
    if len(x_data) >= 2:
        lin_p = np.polyfit(x_data, y_data, 1)
        y_lin_pred = np.polyval(lin_p, x_data)
        linear_rmse_val = rmse(y_data, y_lin_pred)
    else:
        # If only 1 point, linear is basically the same as logistic
        linear_rmse_val = best_rmse_val

    A_opt, B_opt, M_opt, T_opt = best_params
    results.append(
        {
            "seriesNumber": sid,
            "A": A_opt,
            "B": B_opt,
            "M": M_opt,
            "T": T_opt,
            "RMSE_4plogistic": best_rmse_val,
            "RMSE_linear": linear_rmse_val,
            "Diff(linear - 4p)": linear_rmse_val - best_rmse_val,
        }
    )

results_df = pd.DataFrame(results).sort_values("seriesNumber")
print("\n=== FINAL 4-PARAM LOGISTIC RESULTS (ALL SERIES) ===")
print(results_df)

#################################################
# 6. Plot ALL series into logistic_fits.pdf
#################################################
pdf_file = "logistic_fits.pdf"
with PdfPages(pdf_file) as pdf:
    for idx, row in results_df.iterrows():
        sid = row["seriesNumber"]

        A_opt = row["A"]
        B_opt = row["B"]
        M_opt = row["M"]
        T_opt = row["T"]

        grp = df[df["seriesNumber"] == sid]
        x_data = grp["Year"].values
        y_data = grp["Value"].values

        # Prepare logistic curve
        x_min, x_max = x_data.min(), x_data.max()
        if x_min == x_max:
            # single vertical line => just plot points
            fig = plt.figure()
            plt.scatter(x_data, y_data, label="Data")
            plt.title(f"Series {sid} (no horizontal range)")
            pdf.savefig(fig)
            plt.close(fig)
            continue

        x_plot = np.linspace(x_min, x_max, 300)
        y_4p = logistic_4p(x_plot, A_opt, B_opt, M_opt, T_opt)

        # Linear
        if len(x_data) >= 2:
            lin_p = np.polyfit(x_data, y_data, 1)
            y_lin_plot = np.polyval(lin_p, x_plot)
        else:
            # single point => constant line
            y_lin_plot = np.ones_like(x_plot) * y_data[0]

        fig = plt.figure()
        plt.scatter(x_data, y_data, label="Data")
        plt.plot(x_plot, y_4p, label="4P Logistic Fit")
        plt.plot(x_plot, y_lin_plot, "--", label="Linear Fit")

        plt.title(
            f"Series {sid}\n"
            f"4P Logistic RMSE={row['RMSE_4plogistic']:.4f}, "
            f"Linear RMSE={row['RMSE_linear']:.4f}"
        )
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()

        pdf.savefig(fig)
        plt.close(fig)

print(f"\nAll series plotted in {pdf_file}. Done.")
