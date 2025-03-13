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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path
from itertools import cycle


def extract_yearly_energy_from_multiple_dfs(dfs_columns_mapping: dict) -> pd.DataFrame:
    # Initialize an empty dictionary to store the yearly results per column
    yearly_results = {}

    # Iterate over each dataframe and its relevant columns
    for df_name, (df, columns) in dfs_columns_mapping.items():
        # Convert the 'datetime_utc' to datetime if it's not already
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
        df["year"] = df["datetime_utc"].dt.year

        for col in columns:
            # Create a unique column name by combining df_name and col
            col_name = f"{col}"

            # Replace NaN values with zero in the relevant column
            df[col] = df[col].fillna(0)

            # Calculate the total energy consumption for each year
            yearly_energy_consumption = {}
            for year, group in df.groupby("year"):
                first_value = group[col].iloc[0]
                last_value = group[col].iloc[-1]
                total_energy = last_value - first_value
                yearly_energy_consumption[year] = total_energy

            # Store the result in the dictionary under the unique column name
            yearly_results[col_name] = yearly_energy_consumption

    # Create a DataFrame from the results dictionary
    result_df = pd.DataFrame(yearly_results)

    # Ensure that years are in the index and sorted
    result_df.index.name = "Year"
    result_df = result_df.sort_index()

    # Filter out rows where all values are zero
    result_df = result_df.loc[(result_df != 0).any(axis=1)]  # pylint: disable=no-member

    return result_df


def plot_yearly_trends(
    df: pd.DataFrame,
    column_info: dict,
    store_in_file: Path = None,
    all_in_one: bool = True,
    fixed_y_limits: bool = False,
):
    num_columns = df.shape[1]  # Number of columns in the dataframe

    if all_in_one:
        # Create a 2x3 grid for subplots
        fig, axes = plt.subplots(
            2, 3, figsize=(6.9, 4), dpi=400
        )  # Adjust figure size as needed
        axes = (
            axes.ravel()
        )  # Flatten the 2x3 array of axes to make it easier to loop over
        fig.tight_layout(pad=3.0)  # Adjust layout for better spacing

        color_cycle = cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )  # Get the default colors

        # Plot each column on a separate subplot
        for i, col in enumerate(df.columns):
            title, ylabel = column_info.get(
                col, (col, col)
            )  # Get title and ylabel from the dictionary

            # Get the next color from the cycle
            color = next(color_cycle)

            axes[i].bar(df.index, df[col], color=color)  # Bar plot for current column
            axes[i].set_title(title)
            axes[i].set_xlabel("Year")
            axes[i].set_ylabel(ylabel)

            # Set y-axis limits based on the column values
            if fixed_y_limits:
                abs_lim_y = 2800
                if df[col].max() > 0:
                    axes[i].set_ylim(0, abs_lim_y)
                elif df[col].min() < 0:
                    axes[i].set_ylim(-abs_lim_y, 0)

            # Ensure that there are ticks for each bar on the x-axis
            axes[i].set_xticks(df.index)
            axes[i].tick_params(axis="x", rotation=45)  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to prevent overlap

        # Save the plot as both PNG and PDF
        plt.savefig(f"{store_in_file}.png", format="png")
        plt.savefig(f"{store_in_file}.pdf", format="pdf")
        plt.show()

    else:
        # Plot each column individually
        for col in df.columns:
            title, ylabel = column_info.get(
                col, (col, col)
            )  # Fallback to column name if not in dictionary

            plt.figure()
            plt.bar(df.index, df[col])  # Bar plot for current column
            plt.title(title)
            plt.xlabel("Year")
            plt.ylabel(ylabel)
            plt.xticks(
                df.index, rotation=45
            )  # Rotate x-axis labels for better readability
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.show()


parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

with open(parent_dir / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]
    dir_series_reduced = cfg["reduced_datapath"]

plt.style.use(parent_dir / "style.mplstyle")
figure_path = parent_dir.joinpath("figures")

sample_size = "1h"
store_dir_reduced = Path(dir_series_reduced) / f"{sample_size}"

if __name__ == "__main__":
    df_cooling = pd.read_csv(
        os.path.join(store_dir_reduced, f"cooling_W.csv.gz"), compression="gzip"
    )
    df_heating = pd.read_csv(
        os.path.join(store_dir_reduced, f"heating_W.csv.gz"), compression="gzip"
    )
    df_heating.rename(columns={"total": "total_heat_production"}, inplace=True)

    df_electrical = pd.read_csv(
        os.path.join(store_dir_reduced, f"electricity_W.csv.gz"), compression="gzip"
    )
    df_weather = pd.read_csv(
        os.path.join(store_dir_reduced, f"weather.csv.gz"), compression="gzip"
    )

    # fix nans
    df_electrical = df_electrical.fillna(0)
    df_electrical["total_consumption_W"] = (
        df_electrical["total"] - df_electrical["PV"] - df_electrical["CHP"]
    )

    # mapping

    # Define the dataframes and relevant columns in a dictionary
    dfs_columns_mapping = {
        "electrical_total": (df_electrical, ["total_consumption_W"]),
        "electrical_grid": (df_electrical, ["total"]),
        "electrical_PV": (df_electrical, ["PV"]),
        "electrical_CHP": (df_electrical, ["CHP"]),
        "heating_total": (df_heating, ["total_heat_production"]),
        "cooling": (df_cooling, ["cool_elec"]),
        # 'weather': (df_weather, ['weather_temp'])  # You can specify relevant columns here
    }

    # collect energy use/production statistics
    yearly_energy_per_df = extract_yearly_energy_from_multiple_dfs(dfs_columns_mapping)

    for col in yearly_energy_per_df.columns:
        yearly_energy_per_df[col] = yearly_energy_per_df[col] / 1000

    # Accessing the result for a specific dataframe
    print(yearly_energy_per_df)

    # use absolute values for plotting
    yearly_energy_per_df = yearly_energy_per_df.abs()

    # map column names to labels and units
    column_info = {
        "total_consumption_W": ("Total Electrical Consumption", "Energy (MWh)"),
        "total": ("Electrical Energy from Grid", "Energy (MWh)"),
        "PV": ("PV Energy Production", "Energy (MWh)"),
        "CHP": ("CHP Electrical Energy Production", "Energy (MWh)"),
        # "CHP_heat": ("CHP Heat Production", "Energy (MWh)"),
        "total_heat_production": ("Total Heat Production", "Energy (MWh)"),
        "cool_elec": ("Cooling Energy Consumption", "Energy (MWh)"),
    }

    try:
        figure_path.mkdir(exist_ok=True, parents=True)
    except:
        print(f"Could not create figure output directory {figure_path}")
        exit()

    # Assuming 'yearly_energy_df' is your dataframe with years as the index
    plot_yearly_trends(
        yearly_energy_per_df,
        column_info,
        store_in_file=figure_path / "energy_over_years_overview",
    )

    # print data in tabular format
    latex_code = yearly_energy_per_df.to_latex(index=True, float_format="%.2f")
    print(latex_code)

    latex_code_transposed = yearly_energy_per_df.transpose().to_latex(
        index=True, float_format="%.2f"
    )
    print(latex_code_transposed)
