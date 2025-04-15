import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------
# Logistic 3P function
# ---------------------------
def logistic_3p(x, A, M, T):
    return A / (1.0 + np.exp(-M * (x - T)))


# ---------------------------
# Fitting function
# ---------------------------
def fit_logistic_3p(x_data, y_data):
    A_guess = np.max(y_data)
    half_guess = A_guess / 2.0

    try:
        idx_half = (np.abs(y_data - half_guess)).argmin()
        T_guess = x_data[idx_half] if idx_half < len(x_data) else np.median(x_data)
    except:
        T_guess = np.median(x_data)
    M_guess = 1.0

    p0 = [A_guess, M_guess, T_guess]

    try:
        popt, pcov = curve_fit(logistic_3p, x_data, y_data, p0=p0, maxfev=10000)
        y_pred = logistic_3p(x_data, *popt)
        rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))
    except RuntimeError:
        popt, pcov = None, None
        rmse = np.inf

    return popt, pcov, rmse


def fit_linear(x_data, y_data):
    slope, intercept = np.polyfit(x_data, y_data, 1)
    y_pred = slope * x_data + intercept
    rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))
    return slope, intercept, rmse


def main():
    # 1) Read data
    df = pd.read_csv("problem_fits_data_reduced.csv")
    pdf_filename = "my_fits_output.pdf"

    with PdfPages(pdf_filename) as pdf:
        for series_num, group in df.groupby("seriesNumber"):
            x_data = group["Year"].values
            y_data = group["Value"].values

            # Logistic 3P fit
            popt_log, pcov_log, rmse_log = fit_logistic_3p(x_data, y_data)
            if popt_log is not None:
                A_fit, M_fit, T_fit = popt_log
                A_err, M_err, T_err = np.sqrt(np.diag(pcov_log))
            else:
                A_fit = M_fit = T_fit = A_err = M_err = T_err = np.nan

            # Linear fit
            slope, intercept, rmse_lin = fit_linear(x_data, y_data)

            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data", marker="o")

            x_fit_smooth = np.linspace(x_data.min(), x_data.max(), 300)
            if popt_log is not None:
                y_fit_log = logistic_3p(x_fit_smooth, A_fit, M_fit, T_fit)
                ax.plot(
                    x_fit_smooth, y_fit_log, label=f"3P Logistic (RMSE={rmse_log:.3f})"
                )
            else:
                ax.plot([], [], label="Logistic Fit Failed")

            y_fit_lin = slope * x_fit_smooth + intercept
            ax.plot(
                x_fit_smooth,
                y_fit_lin,
                linestyle="--",
                label=f"Linear (RMSE={rmse_lin:.3f})",
            )

            ax.set_title(f"Series {series_num} - Data & Fits")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()

            if popt_log is not None:
                text_str = (
                    f"3P Logistic params:\n"
                    f"A = {A_fit:.4f} ± {A_err:.4f}\n"
                    f"M = {M_fit:.4f} ± {M_err:.4f}\n"
                    f"T = {T_fit:.4f} ± {T_err:.4f}"
                )
                ax.text(
                    0.05, 0.95, text_str, transform=ax.transAxes, va="top", fontsize=9
                )

            pdf.savefig(fig)
            plt.close(fig)

    print(f"All plots saved to '{pdf_filename}'.")


if __name__ == "__main__":
    main()
