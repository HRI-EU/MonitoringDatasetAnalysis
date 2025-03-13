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


def select_urns(choice, meters, weather=False):
    if choice == ["all"]:
        selected_urns = [x for k, v in meters.items() for x in v]
        selected_urns.remove("WeatherStation.Weather")
    elif choice == ["mains"]:
        selected_urns = meters["electricity_main"]
    elif choice == ["electricity"]:
        selected_urns = (
            meters["electricity_main"]
            + meters["local_generators"]
            + meters["servers"]
            + meters["ventilation"]
            + meters["workshops"]
            + meters["emission_lab"]
        )
    elif choice == ["cooling"]:
        selected_urns = meters["cooling"]
    elif choice == ["heating"]:
        selected_urns = meters["heating"]
    elif choice == ["weather"]:
        selected_urns = meters["weather"]
    else:
        selected_urns = choice
    # Include weather
    if weather and choice != ["weather"]:
        selected_urns += ["WeatherStation.Weather"]
    print(f"{len(selected_urns)} selected URNS: {selected_urns}")
    return selected_urns


def select_measurement(measurement, weather_measurement, cleaning, sampling):
    # Define pattern to search in the files

    # Sampling only works with 'corrected' cleaning style
    if cleaning != "corrected_resampled":
        sampling = None

    if measurement == ["all"]:
        urn_pattern = ["_".join(filter(None, (cleaning, sampling))) + ".csv.gz"]
    else:
        urn_pattern = [
            "_".join(filter(None, (x, cleaning, sampling))) + ".csv.gz"
            for x in measurement
        ]

    if weather_measurement == ["all"]:
        weather_pattern = ["_".join(filter(None, (cleaning, sampling))) + ".csv.gz"]
    else:
        weather_pattern = [
            "_".join(filter(None, (x, cleaning, sampling))) + ".csv.gz"
            for x in weather_measurement
        ]
    return urn_pattern, weather_pattern


def create_df(datapath, selected_urns, urn_pattern, weather_pattern):
    # TODO: add progression bar and eta
    df_list = []
    # Iterate over folder
    for urn in selected_urns:
        urnpath = os.path.join(datapath, urn)
        files = os.listdir(urnpath)
        if urn == "WeatherStation.Weather":
            selected_files = [
                f for f in files for x in weather_pattern if f.endswith(x)
            ]
        else:
            selected_files = [f for f in files for x in urn_pattern if f.endswith(x)]
        # Read data
        for file in selected_files:
            print(f"Reading: {file}")
            df_tmp = pd.read_csv(
                os.path.join(datapath, urn, file),
                compression="gzip",  # nrows=1000,
            )
            df_tmp["datetime_utc"] = pd.to_datetime(df_tmp["datetime_utc"])
            df_tmp.set_index(["datetime_utc"], inplace=True)
            df_tmp = df_tmp.tz_convert(None)
            df_list.append(df_tmp)
    df = pd.concat(df_list, axis=1)
    print("Reading done!")
    return df


def save_file(
    df,
    urns,
    measurement,
    weather,
    weather_measurement,
    cleaning_type,
    sampling_type,
    outpath,
):
    filename = (
        "_".join(
            (
                [
                    urns[0]
                    if urns
                    in (["all"], ["cold"], ["heat"], ["electricity"], ["weather"])
                    else "various"
                ][0],
                "".join(measurement),
                ["".join(weather_measurement) if weather else "none"][0],
                cleaning_type,
                sampling_type,
            )
        )
        + ".csv.gz"
    )

    os.makedirs(outpath, exist_ok=True)
    print(f"Saving compressed csv to: {os.path.join(outpath, filename)}")
    df.to_csv(os.path.join(outpath, filename), compression="gzip")
    print("DONE!")
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script reads the MonitoringServerData already downloaded, "
        "and it creates a single dataframe with selected urns and measurements. "
        "Output file will be compressed by gzip. Name structure: "
        "<urns>_<meters>_<weather meters>_<cleaning>_<sampling>.csv.gz"
    )

    # In windows, have to mount the path on a virtual drive.
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        default=r"Y:\MonitoringServerDataSet\2024_10_DMD2_puslished_data\data",
        help="Path where data is.",
    )

    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="../data/",
        help="Path where to save the output file. Default: ../data/",
    )

    parser.add_argument(
        "-u",
        "--urns",
        nargs="+",
        type=str,
        default=["cold"],
        help="URNs to consider. Choices: 'all', 'electricity', 'heating', 'cooling', 'weather', or sequence of single URNs e.g., H1.Z20 V.Z81 etc...",
    )  # define subset of urns
    parser.add_argument(
        "-m",
        "--measurement",
        nargs="+",
        type=str,
        default=["all"],
        help="'all' or single meter e.g. 'P', 'W' etc...",
    )
    parser.add_argument(
        "-w",
        "--weather",
        action="store_true",
        default=False,
        help="Includes the weather data",
    )
    parser.add_argument(
        "-wm",
        "--weather_measurement",
        nargs="+",
        type=str,
        default=["Ta"],
        help="'all' or single meter e.g. 'Ta' etc...",
    )
    parser.add_argument(
        "-c",
        "--cleaning_type",
        type=str,
        default="corrected_resampled",
        choices=("raw", "harmonized", "corrected", "corrected_resampled"),
    )
    parser.add_argument(
        "-s",
        "--sampling_type",
        type=str,
        default="1min",
        choices=("1min", "15min", "1h"),
        help="This is valid only if 'resampled' cleaning type is selected",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Checks
    # TODO: do more checks
    if args.cleaning_type in ("raw", "normalized"):
        assert args.sampling_type == "raw"

    with open(f"meters.yaml", "r") as f:
        meters = yaml.safe_load(f)

    all_meters = [x for k, v in meters.items() for x in v]
    print(f"Total URNs: {len(all_meters)}")

    selected_urns = select_urns(args.urns, meters, args.weather)
    urn_pattern, weather_pattern = select_measurement(
        args.measurement,
        args.weather_measurement,
        args.cleaning_type,
        args.sampling_type,
    )
    df = create_df(args.datapath, selected_urns, urn_pattern, weather_pattern)

    # save
    # save_file(df, args.urns, args.measurement, args.weather, args.weather_measurement,
    #           args.cleaning_type, args.sampling_type, args.outpath)

    ##
    # import numpy as np
    # df.replace('ffill', np.nan, inplace=True)
    # df.replace('bfill', np.nan, inplace=True)
    # df.replace('nan', np.nan, inplace=True)
    # df_nan = df.isna()
    # df_nan.index = df_nan.index.strftime("%d/%m/%y %H:%M")
    #
    # fig, ax = plt.subplots(figsize=(19.8, 20))
    # sns.heatmap(df_nan.T, cbar=False)
    # plt.savefig(os.path.join(PATH, filename.split(sep='.')[0]) + '_missing', dpi=200)
    # print("Done! Figure in:", os.path.join(PATH, filename.split(sep='.')[0]) + '_missing')
