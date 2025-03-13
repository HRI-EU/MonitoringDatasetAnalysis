#
# Copyright (c) 2025, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from datetime import datetime
import os
import sys

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import glob
import yaml

from pathlib import Path

# if true, evaluation plots for all meters will be created
verbose_plotting = False

# if true, main statistics plots will be stored (if false, no plots at all will be stored)
store_plots = True

SAMPLING = "1min"


def read_P_W(urn: str, pre_path: str = "."):
    try:
        df_P = pd.read_csv(
            pre_path
            + os.sep
            + urn
            + os.sep
            + urn
            + f".P_corrected_resampled_{SAMPLING}.csv.gz",
            compression="gzip",
        )
    except Exception as e:
        print(getattr(e, "message", str(e)))
        error_out.write(getattr(e, "message", str(e)) + "\n")
        return None

    try:
        df_W = pd.read_csv(
            pre_path
            + os.sep
            + urn
            + os.sep
            + urn
            + f".W_corrected_resampled_{SAMPLING}.csv.gz",
            compression="gzip",
        )
    except Exception as e:
        print(getattr(e, "message", str(e)))
        error_out.write(getattr(e, "message", str(e)) + "\n")
        return None

    df_P["datetime_utc"] = pd.to_datetime(df_P["datetime_utc"], utc=True)
    df_P.set_index(["datetime_utc"], inplace=True)

    df_W["datetime_utc"] = pd.to_datetime(df_W["datetime_utc"], utc=True)
    df_W.set_index(["datetime_utc"], inplace=True)

    # get array of timestamps
    time_P = (
        (df_P.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    ).to_numpy()
    time_W = (
        (df_W.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    ).to_numpy()

    if (
        len(time_P) != len(time_W)
        or (time_P[0] != time_W[0])
        or (time_P[-1] != time_W[-1])
    ):
        print(
            f"WARNING: {urn}: different sizes of time arrays or start and/or end points dont match! => correct"
        )
        # get timestamps in both time series
        overlapping_time, time_P_idx, time_W_idx = np.intersect1d(
            time_P, time_W, assume_unique=True, return_indices=True
        )

    else:

        overlapping_time = time_P
        time_P_idx = [0, len(time_P) - 1]
        time_W_idx = [0, len(time_P) - 1]

    numT = len(overlapping_time)

    P = df_P[urn + ".P"].to_numpy()[
        time_P_idx[0] : time_P_idx[-1] + 1
    ]  ## need the +1 since last index is usually excluded
    W = df_W[urn + ".W"].to_numpy()[time_W_idx[0] : time_W_idx[-1] + 1]

    if (len(P) != numT) or (len(W) != numT):
        print(
            f"ERROR {urn}: time and measurement series dont match. W:{len(W)}, P:{len(P)}, time:{numT}!"
        )
        error_out.write(
            f"ERROR: {urn}: time and measurement series dont match. W:{len(W)}, P:{len(P)}, time:{numT}\n"
        )
        return None
        sys.exit(8)

    nan_P = np.where(np.isnan(P))[0]
    nan_W = np.where(np.isnan(W))[0]
    if (len(nan_P) > 0) or (len(nan_W) > 0):
        print(f"{urn}: series contain nans: w:{len(nan_W)}, P:{len(nan_P)}!")
        error_out.write(
            f"ERROR: {urn}: series contain nans: W:{len(nan_W)}, P:{len(nan_P)}!\n"
        )
        print("#nan P", nan_P)
        print("#nan W", nan_W)

        print("Ps", df_P[np.isnan(P)])
        print("Ws", df_W[np.isnan(W)])
        return None

    return overlapping_time, P, W


def integrate_P(t_arr, P):

    WPs = scipy.integrate.simpson(P, x=t_arr) / 3600000.0  ## transform to kWh from Ws

    return WPs


def differentiate_W(time_arr, W, P, key):
    dW = 3600000.0 * np.diff(W)
    dt = np.diff(time_arr)
    # forward finite difference
    Pd_fd = dW / dt
    # central finite difference
    Pd_cd = 0.5 * (Pd_fd[:-1] + Pd_fd[1:])
    if verbose_plotting:
        # pick random interval
        Nt = 1000
        toff = len(time_arr) - 3 * Nt
        Nt = 240
        toff = len(time_arr) - 2200
        time_arr_plot = (
            pd.to_datetime(time_arr[toff : toff + Nt], unit="s")
            .tz_localize("utc")
            .tz_convert("Europe/Berlin")
        )
        Pscale = np.ptp(P[toff : toff + Nt], axis=0)
        Pmean = np.mean(P[toff : toff + Nt], axis=0)
        plt.figure(figsize=(9, 5))
        plt.tight_layout(pad=1)
        plt.plot(
            time_arr_plot,
            Pd_fd[toff : toff + Nt],
            "r",
            lw=2,
            label=r"${\frac{\Delta W}{\Delta t}}_\mathrm{forward}$",
        )
        plt.plot(
            time_arr_plot,
            Pd_cd[toff : toff + Nt],
            "b",
            lw=2,
            label=r"$\frac{\Delta W}{\Delta t}_\mathrm{central}$",
        )
        plt.plot(time_arr_plot, P[toff : toff + Nt], "k", lw=2, label=r"measured $P$")
        plt.title(f"{key}")
        plt.legend(fontsize=18)
        plt.savefig(verbose_figure_path / f"{key}_dWdt.png")
        # plt.show()
    return Pd_cd


parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

with open(parent_dir / "config.yaml", "r") as file:
    cfg = yaml.safe_load(file)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]
    dir_series_reduced = cfg["reduced_datapath"]

plt.style.use(parent_dir / "style.mplstyle")
figure_path = parent_dir.joinpath("figures")
verbose_figure_path = figure_path.joinpath("statistics_p_vs_w")

if __name__ == "__main__":
    if store_plots:
        figure_path.mkdir(exist_ok=True)
        if verbose_plotting:
            verbose_figure_path.mkdir(exist_ok=True)

    log_dir = parent_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    error_out = open(log_dir / "p_vs_w_time_series_interval.log", "w")

    pre_path = dir_series

    file_list = [os.path.basename(fl.rstrip("/")) for fl in glob.glob(pre_path + "/*/")]
    print("try to read time series\n", file_list)

    numTimeSeries = len(file_list)
    if not file_list:
        print("no files found!")
        sys.exit(9)

    of = open(log_dir / "statistics_p_vs_w.txt", "w")
    of.write("#key W_measured integral_P error_W[%] P_norm error_P error_P_rel[%]\n")

    minutes_per_day = 24 * 60

    errors_all = []
    errors_dP = []

    W_err_global = 0.0
    P_err_global = 0.0

    for index, file in enumerate(file_list):
        #
        error_out.write("#\n")
        error_out.flush()
        print(f"\n# {file}")
        sys.stdout.flush()

        # read time arra and Power  values
        read_series = read_P_W(file, pre_path)
        if read_series is None:
            print(f"{file} error encounter -> skip.")
            error_out.write(f"{file} error encounter -> skip.\n")
            numTimeSeries -= 1  # reduce to get correct normalization
            continue

        time_arr, P_meas, W_meas = read_series
        start_sec = datetime.fromtimestamp(time_arr[0]).second

        if start_sec != 0:
            print(f"({file}: not full minute: {datetime.fromtimestamp(time_arr[0])} ")
            error_out.write(
                f"({file}: not full minute: {datetime.fromtimestamp(time_arr[0])}\n"
            )

        W_integrated = integrate_P(time_arr, P_meas)
        W_meas_tot = W_meas[-1] - W_meas[0]
        W_err_total = 100 * np.abs((W_integrated / W_meas_tot - 1.0))

        W_err_global += W_err_total

        dW_dt = differentiate_W(time_arr, W_meas, P_meas, file)
        err_P = np.mean(np.abs(dW_dt - P_meas[1:-1]))
        normP = np.mean(np.abs(P_meas[1:-1]))
        print("pint", err_P, normP)
        err_P_rel = 100 * (err_P / normP)

        P_err_global += err_P_rel

        print(f"# Perr[%] {err_P_rel}")

        of.write(
            f"{file} {W_meas_tot:.10g} {W_integrated} {W_err_total} {normP} {err_P} {err_P_rel}\n"
        )

        Ndays = len(time_arr) // minutes_per_day
        errors = np.zeros(Ndays - 1)

        print(
            f"Info: {file} num_t:{len(time_arr)} ({Ndays} days) interval: {datetime.fromtimestamp(time_arr[0])} - {datetime.fromtimestamp(time_arr[-1])}"
        )
        error_out.write(
            f"Info: {file} num_t:{len(time_arr)} ({Ndays} days) interval: {datetime.fromtimestamp(time_arr[0])} - {datetime.fromtimestamp(time_arr[-1])}\n"
        )

        of.flush()

        for ni in range(Ndays - 1):
            W_integrated_n = integrate_P(
                time_arr[ni * minutes_per_day : (ni + 1) * minutes_per_day],
                P_meas[ni * minutes_per_day : (ni + 1) * minutes_per_day],
            )
            W_meas_n = (
                W_meas[(ni + 1) * minutes_per_day - 1] - W_meas[ni * minutes_per_day]
            )
            Pscale_n = np.sqrt(
                np.mean(P_meas[ni * minutes_per_day : (ni + 1) * minutes_per_day] ** 2)
            )
            Wscale_n = np.sqrt(
                np.mean(
                    (
                        W_meas[ni * minutes_per_day : (ni + 1) * minutes_per_day - 1]
                        - W_meas[ni * minutes_per_day]
                    )
                    ** 2
                )
            )
            Enorm = np.maximum(Wscale_n, Pscale_n)
            if Enorm < 1.0:

                Enorm = 1.0
            erri = 100 * (W_integrated_n - W_meas_n) / Enorm
            errMP = 0.7
            errMN = -0.9
            if erri > errMP:

                erri = errMP
            elif erri < errMN:

                erri = errMN

            errors[ni] = erri

            ## dW / dt
            ## regulate derivative if no change detected in P -> avoid division by 0
            denominator = np.maximum(
                np.sum(
                    np.abs(
                        P_meas[
                            1 + ni * minutes_per_day : 1 + (ni + 1) * minutes_per_day
                        ]
                    )
                ),
                1.0e-6,
            )
            errDP = (
                100
                * np.sum(
                    np.abs(
                        dW_dt[ni * minutes_per_day : (ni + 1) * minutes_per_day]
                        - P_meas[
                            1 + ni * minutes_per_day : 1 + (ni + 1) * minutes_per_day
                        ]
                    )
                )
                / denominator
            )
            if errDP > 170:

                errDP = 170.0
            errors_dP.append(errDP)

        # print('days done')
        sys.stdout.flush()
        errors_all.append(errors)

        if verbose_plotting:
            plt.figure(figsize=(10, 8))
            plt.hist(
                errors, bins=301
            )  # bins='auto')  # arguments are passed to np.histogram
            plt.title(f"{file}")
            plt.ylabel("count")
            plt.xlabel(r"$\Delta W$ [%]")
            plt.yscale("log")
            plt.savefig(verbose_figure_path / f"delta_W_{file}.png")
            plt.tight_layout(pad=1)

    print("all URNs done")
    W_err_global /= numTimeSeries
    P_err_global /= numTimeSeries
    print(f"Global Error W: {W_err_global}%")
    print(f"Global Error P: {P_err_global}%")
    of.write(f"## Average Global Error W[%]: {W_err_global}\n")
    of.write(f"## Average Global Error P[%]: {P_err_global}\n")
    sys.stdout.flush()
    of.close()

    errors_all = np.concatenate(errors_all, axis=0)
    plt.figure(figsize=(5, 5))
    plt.hist(errors_all, bins=601)  # 'auto'  # arguments are passed to np.histogram
    # plt.title(f"All ")
    # plt.text()
    plt.ylabel("count")
    plt.xlabel(r"$\Delta W$ [%]")
    plt.yscale("log")
    if store_plots:
        plt.savefig(figure_path / f"delta_W_All.pdf")
    plt.tight_layout(pad=1)

    errors_dP = np.asarray(errors_dP)
    # fig, ax1 = plt.subplots()
    plt.figure(figsize=(5, 5))
    plt.hist(errors_dP, bins=601)  # 'auto'  # arguments are passed to np.histogram
    # plt.title(f"All ")
    # plt.text()
    plt.ylabel("count")
    plt.xlabel(r"$|\Delta P|$ [%]")
    plt.yscale("log")
    if store_plots:
        plt.savefig(figure_path / f"delta_P_All.pdf")
    plt.tight_layout(pad=1)

    error_out.close()
    # plt.show()
