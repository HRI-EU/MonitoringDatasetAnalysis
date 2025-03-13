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

import os
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from ReadFiles import select_urns, select_measurement, create_df

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script creates the reduced datasets from the MonitoringSystem Data"
    )

    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--cleaning_type",
        type=str,
        default="corrected_resampled",
        choices=("raw", "corrected_resampled", "harmonized", "corrected"),
    )
    parser.add_argument(
        "-s",
        "--sampling_type",
        type=str,
        default="1h",
        choices=("raw", "1min", "15min", "1h"),
    )
    args = parser.parse_args()
    return args


def helper_get_data(meter_selected, meters, variable, weather, correction, sampling):
    selected_urns = select_urns(meter_selected, meters, weather=weather)
    urn_pattern, weather_pattern = select_measurement(
        variable, ["Ta", "Igm"], correction, sampling
    )
    df = create_df(datapath, selected_urns, urn_pattern, weather_pattern)
    return df


if __name__ == "__main__":
    args = parse_args()

    CORRECTION = args.cleaning_type
    SAMPLING = args.sampling_type

    outpath = Path(args.out_path)

    parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

    with open(parent_dir / "meters.yaml", "r") as f:
        meters = yaml.safe_load(f)
    with open(parent_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    datapath = cfg["datapath"]

    outpath = outpath / "reduced_data"
    if outpath.exists():
        print(f"Warning: output path already exists at {outpath}")

    try:
        outpath.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        print(f"Could not create output directory {outpath}:\n{ex}")

    outpath = os.path.join(outpath, "_".join([CORRECTION, SAMPLING]))
    os.makedirs(outpath, exist_ok=True)
    print(f"{datapath=}")
    print(f"{outpath=}")
    all_meters = [x for k, v in meters.items() for x in v]

    # Electricity P
    df = helper_get_data(["mains"], meters, ["P"], False, CORRECTION, SAMPLING)
    main_elec_P = df.sum(axis=1)
    df_elec_P = pd.DataFrame({"total": main_elec_P})
    # PV P
    df = helper_get_data(
        ["V.Z84", "H1.Z310", "H2.Z311", "H3.Z312"],
        meters,
        ["P"],
        False,
        CORRECTION,
        SAMPLING,
    )
    PV_P = df.sum(axis=1)
    df_elec_P["PV"] = PV_P
    # CHP P
    df_chp_p = helper_get_data(["H1.Z20"], meters, ["P"], False, CORRECTION, SAMPLING)
    df_elec_P["CHP"] = df_chp_p
    df_elec_P = df_elec_P[:"2023-12-31 22:59:00"]
    df_elec_P.ffill(inplace=True)
    df_elec_P.to_csv(
        os.path.join(outpath, f"electricity_P.csv.gz"), sep=",", compression="gzip"
    )

    # Electricity W
    df = helper_get_data(["mains"], meters, ["W"], False, CORRECTION, SAMPLING)
    df.bfill(inplace=True)
    df -= df.iloc[0]
    df.ffill(inplace=True)
    main_elec_W = df.sum(axis=1)
    df_elec_W = pd.DataFrame({"total": main_elec_W})
    # PV P
    df = helper_get_data(
        ["V.Z84", "H1.Z310", "H2.Z311", "H3.Z312"],
        meters,
        ["W"],
        False,
        CORRECTION,
        SAMPLING,
    )
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df -= df.iloc[0]
    PV_W = df.sum(axis=1)
    df_elec_W["PV"] = PV_W
    # CHP P
    df_chp_w = helper_get_data(["H1.Z20"], meters, ["W"], False, CORRECTION, SAMPLING)
    df_chp_w.bfill(inplace=True)
    df_chp_w -= df_chp_w.iloc[0]
    df_elec_W["CHP"] = df_chp_w
    df_elec_W = df_elec_W[:"2023-12-31 22:59:00"]
    df_elec_W.ffill(inplace=True)
    df_elec_W.to_csv(
        os.path.join(outpath, f"electricity_W.csv.gz"), sep=",", compression="gzip"
    )

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), sharex=True)
    df_elec_P.plot(ax=axes[0], alpha=0.75)
    df_elec_W.plot(ax=axes[1], alpha=0.75)
    axes[0].set_ylabel("Power (W)")
    axes[1].set_ylabel("Energy (kWh)")
    # axes[0].set_ylim([-1e6, 1e6])
    # axes[1].set_ylim([-5e6, 9e6])
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"electricity"), dpi=300, bbox_inches="tight")

    ############################################################################################################
    # Heating
    # P
    df = helper_get_data(
        ["H1.W11", "H1.W12"], meters, ["P"], False, CORRECTION, SAMPLING
    )
    df_heat_P = pd.DataFrame(
        {
            "total": df["H1.W11.P"],
            "CHP_heat": df["H1.W12.P"],
            "CHP_elec": df_chp_p["H1.Z20.P"],
        }
    )
    df_heat_P = df_heat_P[:"2023-12-31 22:59:00"]
    df_heat_P.ffill(inplace=True)
    df_heat_P.to_csv(
        os.path.join(outpath, f"heating_P.csv.gz"), sep=",", compression="gzip"
    )

    # W
    df = helper_get_data(
        ["H1.W11", "H1.W12"], meters, ["W"], False, CORRECTION, SAMPLING
    )
    df_heat_W = pd.DataFrame(
        {
            "total": df["H1.W11.W"],
            "CHP_heat": df["H1.W12.W"],
            "CHP_elec": df_chp_w["H1.Z20.W"],
        }
    )
    df_heat_W.bfill(inplace=True)
    df_heat_W.ffill(inplace=True)
    df_heat_W -= df_heat_W.iloc[0]
    df_heat_W = df_heat_W[:"2023-12-31 22:59:00"]
    df_heat_W.ffill(inplace=True)
    df_heat_W.to_csv(
        os.path.join(outpath, f"heating_W.csv.gz"), sep=",", compression="gzip"
    )

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), sharex=True)
    df_heat_P.plot(ax=axes[0], alpha=0.75)
    df_heat_W.plot(ax=axes[1], alpha=0.75)
    axes[0].set_ylabel("Power (kW)")
    axes[1].set_ylabel("Energy (kWh)")
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"heating"), dpi=300, bbox_inches="tight")

    ############################################################################################################
    # Cooling
    # P
    df = helper_get_data(
        ["V.K21", "H1.Z16", "H1.Z11", "H1.Z12", "H1.Z24", "H1.Z25"],
        meters,
        ["P"],
        False,
        CORRECTION,
        SAMPLING,
    )
    df_cool_P = pd.DataFrame(
        {
            "total": df["V.K21.P"],
            "cool_elec": df[
                ["H1.Z16.P", "H1.Z11.P", "H1.Z12.P", "H1.Z24.P", "H1.Z25.P"]
            ].sum(axis=1),
        }
    )
    df_cool_P = df_cool_P[:"2023-12-31 22:59:00"]
    df_cool_P.ffill(inplace=True)
    df_cool_P.to_csv(
        os.path.join(outpath, f"cooling_P.csv.gz"), sep=",", compression="gzip"
    )

    # W
    df = helper_get_data(
        ["V.K21", "H1.Z16", "H1.Z11", "H1.Z12", "H1.Z24", "H1.Z25"],
        meters,
        ["W"],
        False,
        CORRECTION,
        SAMPLING,
    )
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df -= df.iloc[0]
    df_cool_W = pd.DataFrame(
        {
            "total": df["V.K21.W"],
            "cool_elec": df[
                ["H1.Z16.W", "H1.Z11.W", "H1.Z12.W", "H1.Z24.W", "H1.Z25.W"]
            ].sum(axis=1),
        }
    )
    df_cool_W = df_cool_W[:"2023-12-31 22:59:00"]
    df_cool_W.ffill(inplace=True)
    df_cool_W.to_csv(
        os.path.join(outpath, f"cooling_W.csv.gz"), sep=",", compression="gzip"
    )

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), sharex=True)
    df_cool_P.plot(ax=axes[0], alpha=0.75)
    df_cool_W.plot(ax=axes[1], alpha=0.75)
    axes[0].set_ylabel("Power (W)")
    axes[1].set_ylabel("Energy (kWh)")
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"cooling"), dpi=300, bbox_inches="tight")

    ############################################################################################################
    # Weather
    df_weather = helper_get_data(
        ["weather"], meters, ["None"], True, CORRECTION, SAMPLING
    )
    df_weather.to_csv(
        os.path.join(outpath, f"weather.csv.gz"), sep=",", compression="gzip"
    )

    df_weather.plot(alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"weather"), dpi=300, bbox_inches="tight")

    ##
    # plt.show()
    plt.close("all")
    print("DONE!")
