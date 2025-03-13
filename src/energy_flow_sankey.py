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
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from ReadFiles import select_urns, select_measurement, create_df

import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# plt.switch_backend("Qt5Agg")

SAMPLING = "15min"
CORR_FACTOR_DICT = {"1h": 1000, "15min": 4000, "1min": 60000}


def helper_get_data(
    meter_selected: list[str],
    meters: dict[str, list[str]],
    variable: str,
    weather: bool,
    correction: str,
    sampling: str,
) -> pd.DataFrame:
    selected_urns = select_urns(meter_selected, meters, weather=weather)
    urn_pattern, weather_pattern = select_measurement(
        variable, ["Ta", "Igm"], correction, sampling
    )
    df = create_df(datapath, selected_urns, urn_pattern, weather_pattern)
    return df


def get_delta_energy(df, start=None, end=None) -> float:
    df_range = df[start:end].copy()
    df_range.bfill(inplace=True)
    df_range.ffill(inplace=True)
    w0 = df_range.iloc[0]
    w1 = df_range.iloc[-1]
    W = w1 - w0
    return W


def do_sankey(
    producer_df: pd.DataFrame,
    consumer_df: pd.DataFrame,
    title: str = "",
    scale: float = 6e-5,
    offset: float = 0.25,
    figsize: tuple = (8, 12),
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title=title)
    ax.axis("off")
    sankey = Sankey(scale=scale, offset=offset, format="%.1f", unit="", ax=ax)
    total_inflows = producer_df["grid_in"] + producer_df["CHP"] + producer_df["PV"]
    sankey.add(
        flows=[
            producer_df["grid_in"],
            producer_df["CHP"],
            producer_df["PV"],
            -total_inflows,
        ],
        labels=["Grid In", "CHP", "PV", None],
        orientations=[-1, 1, 1, 0],  # Control arrow direction
        trunklength=0.7,
        rotation=0,
    )
    sankey.add(
        flows=[
            total_inflows,
            producer_df["grid_out"],
            -(total_inflows + producer_df["grid_out"]),
        ],
        labels=[None, "Grid Out", None],
        orientations=[0, -1, 0],
        prior=0,
        connect=(3, 0),
        trunklength=0.8,
        rotation=0,
    )
    sankey.add(
        flows=[
            total_inflows + producer_df["grid_out"],
            -consumer_df["Office"],
            -consumer_df["Design Studio"],
            -consumer_df["Cooling"],
            -consumer_df["Ventilation"],
            -consumer_df["Workshop"],
            -consumer_df["Emission Lab"],
            -consumer_df["Others"],
        ],
        labels=[
            "Consumption",
            "Office",
            "Design Studio",
            "Cooling",
            "Ventilation",
            "Workshop",
            "Emission Lab",
            "Others",
        ],
        orientations=[0, -1, -1, -1, -1, 1, 1, 1],
        prior=1,
        connect=(2, 0),
        trunklength=0.9,
        rotation=0,
    )
    sankey.finish()
    fig.tight_layout()


if __name__ == "__main__":
    parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    plt.style.use(parent_dir / "style.mplstyle")

    with open(parent_dir / "meters.yaml", "r") as f:
        meters = yaml.safe_load(f)
    with open(parent_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    datapath = cfg["datapath"]

    print(f"{datapath=}")
    all_meters = [x for k, v in meters.items() for x in v]

    corr_factor = CORR_FACTOR_DICT[SAMPLING]

    df_main = helper_get_data(
        ["mains"], meters, ["P"], False, "corrected_resampled", SAMPLING
    )
    df_main = df_main.sum(axis=1)

    main_elec = pd.DataFrame()
    main_elec["grid_out"] = df_main.clip(upper=0)
    main_elec["grid_in"] = df_main.clip(lower=0)

    # pv
    df = helper_get_data(
        ["V.Z84", "H1.Z310", "H2.Z311", "H3.Z312"],
        meters,
        ["P"],
        False,
        "corrected_resampled",
        SAMPLING,
    )
    df.fillna(0, inplace=True)
    df = df.sum(axis=1)
    main_elec["PV"] = df * -1

    df = helper_get_data(
        ["H1.Z20"], meters, ["P"], False, "corrected_resampled", SAMPLING
    )
    main_elec["CHP"] = df * -1

    # Get energy
    W_producer = main_elec.cumsum() / corr_factor  # kWh

    # Consumers - Labs
    df = helper_get_data(
        [
            "H1.Z15",
            "H1.Z28",
            "H2.T.Z32",
            "H2.T.Z33",
            "H2.T.Z30",
            "H2.T.Z34",
            "H1.Z20",
            "H1.Z310",
            "H3.Z312",
            "H1.Z16",
            "H1.Z11",
            "H1.Z12",
            "H1.Z24",
            "H1.Z25",
            "H2.Z68",
            "H2.Z69",
            "H2.Z70",
            "H3.Z42",
        ],
        meters,
        ["P"],
        False,
        "corrected_resampled",
        SAMPLING,
    )
    df.fillna(0, inplace=True)
    df_consumers = pd.DataFrame()
    df_consumers["Emission Lab"] = (
        df["H1.Z15.P"] + df["H1.Z28.P"] - df["H1.Z20.P"] - df["H1.Z310.P"]
    )
    df_consumers["Design Studio"] = df["H2.T.Z33.P"] - df["H3.Z312.P"]
    df_consumers["Office"] = df["H2.T.Z30.P"]
    df_consumers["Workshop"] = df["H2.T.Z34.P"]
    # df_consumers.clip(lower=0)

    # df_other = helper_get_data(['H2.T.Z31', 'H2.T.Z32'], meters, ['P'], False, 'corrected', SAMPLING).sum(axis=1)

    df_consumers["Cooling"] = df[
        ["H1.Z16.P", "H1.Z11.P", "H1.Z12.P", "H1.Z24.P", "H1.Z25.P"]
    ].sum(axis=1)
    df_consumers["Ventilation"] = df[
        ["H2.Z68.P", "H2.Z69.P", "H2.Z70.P", "H3.Z42.P"]
    ].sum(axis=1)

    df_consumers["Emission Lab"] -= df_consumers["Cooling"]
    df_consumers["Workshop"] -= df[["H2.Z68.P", "H2.Z69.P", "H2.Z70.P"]].sum(axis=1)
    df_consumers["Design Studio"] -= df["H3.Z42.P"]

    # df_consumers['Lab'] = df_consumers['Emission Lab'] + df_consumers['Workshop']
    # df_consumers['Offices'] = df_consumers['Office'] + df_consumers['Design Studio']
    # df_consumers.drop(['Emission Lab', 'Workshop', 'Office', 'Design Studio'], axis=1, inplace=True)

    df_consumers["Others"] = main_elec.sum(axis=1) - df_consumers.sum(axis=1)

    df_consumers.clip(lower=0)
    W_consumers = df_consumers.cumsum() / corr_factor

    # Get shit done
    start = None  #'2023-09-18'
    end = None  #'2023-09-20'
    producers = get_delta_energy(W_producer, start, end) / 1e3  # in MWh
    consumer = get_delta_energy(W_consumers, start, end) / 1e3  # in MWh

    tot_energy = producers[["grid_in", "PV", "CHP"]].sum()
    consumption = producers.sum()
    grid_out = -producers["grid_out"]
    c_sum = consumer.sum()
    print(f"check: {consumption=:.0f} : {c_sum=:.0f} (MWh)")
    print(f"diff= {(consumption-c_sum)=:.0f} (MWh)")

    # # Get percentage
    # cp = c/tot_energy
    # pp = p/tot_energy

    figure_path = parent_dir.joinpath("figures")
    try:
        figure_path.mkdir(exist_ok=True, parents=True)
    except:
        print(f"Could not create figure output directory {figure_path}")
        exit()

    # Sankey
    for y in ["2018", "2019", "2020", "2021", "2022", "2023"]:
        try:
            start = end = y
            producers = get_delta_energy(W_producer, start, end) / 1e3  # in MWh
            consumer = get_delta_energy(W_consumers, start, end) / 1e3  # in MWh
            do_sankey(
                producers,
                consumer,
                title=f"Energy Flow in {y} (MWh).",
                scale=3e-4,
                figsize=(8, 12),
            )
            output_file = figure_path / f"sankey_{y}.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Could not create Sankey diagram for year {y}:\n{e}")

    # Full data
    start = end = None
    plt.rcParams.update({"font.size": 9})  # Set the font size to 14 (or any other size)

    producers = get_delta_energy(W_producer, start, end) / 1e3  # in MWh
    consumer = get_delta_energy(W_consumers, start, end) / 1e3  # in MWh

    do_sankey(producers, consumer, title="", scale=4.5e-5, figsize=(7, 3.5), offset=0.2)
    plt.savefig(figure_path / "sankey_full.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    #%% print relative shares used in the text
    percentage_grid = (
        producers["grid_in"]
        / (producers["grid_in"] + producers["PV"] + producers["CHP"])
        * 100
    )
    percentage_chp = (
        producers["CHP"]
        / (producers["grid_in"] + producers["PV"] + producers["CHP"])
        * 100
    )
    percentage_pv = (
        producers["PV"]
        / (producers["grid_in"] + producers["PV"] + producers["CHP"])
        * 100
    )
    percentage_feedin = (
        abs(producers["grid_out"])
        / (producers["grid_in"] + producers["PV"] + producers["CHP"])
        * 100
    )

    print(f"share of grid: {percentage_grid:.0f}%")
    print(f"share of chp: {percentage_chp:.0f}%")
    print(f"share of pv: {percentage_pv:.0f}%")
    print(f"share of feed-in back to grid: {percentage_feedin:.0f}%")

    percentage_emlab = consumer["Emission Lab"] / consumer.sum() * 100
    percentage_workshop = consumer["Workshop"] / consumer.sum() * 100
    percentage_hvac = (
        (consumer["Cooling"] + consumer["Ventilation"]) / consumer.sum() * 100
    )
    percentage_rest = (
        (consumer["Others"] + consumer["Office"] + consumer["Design Studio"])
        / consumer.sum()
        * 100
    )

    print(f"share of emlab + workshop: {percentage_emlab+percentage_workshop:.0f}%")
    print(f"share of hvac: {percentage_hvac:.0f}%")
    print(f"share of rest: {percentage_rest:.0f}%")
