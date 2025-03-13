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
import pandas as pd
import numpy as np
import yaml

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import colorsys
import seaborn as sns


def adjust_lightness(color, amount=0.5):
    if color in matplotlib.colors.cnames:
        c = matplotlib.colors.cnames[color]
    if isinstance(color, str) and (color[0] == "#" or len(color) == 6):  #
        c = matplotlib.colors.ColorConverter.to_rgb(color)
    else:
        c = color

    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


#%%
def plot_timeseries(
    dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    alpha: float = 1,
    linewidth: float = 1,
) -> (plt.Figure, np.ndarray[plt.Axes]):
    elec, cool, heat, weather = dataframes

    legend_loc = "upper left"

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6.9, 2 * 2), dpi=300)
    elec["load"] = elec.total - elec.PV.fillna(0) - elec.CHP
    (elec.total / 1000).plot(
        ax=axes[0],
        label="Grid demand",
        alpha=alpha,
        linewidth=linewidth,
        color=colors_array[0],
    )
    (elec.PV / 1000).plot(
        ax=axes[0],
        label="PV electrical production",
        alpha=alpha,
        linewidth=linewidth,
        color=colors_array[1],
    )
    (elec.CHP / 1000).plot(
        ax=axes[0],
        label="CHP electrical production",
        alpha=alpha,
        linewidth=linewidth,
        color=colors_array[2],
    )
    legend = axes[0].legend(loc="upper left", ncols=3)
    for handle in legend.legend_handles:
        handle.set_linewidth(1.5)

    axes[0].set_title("Electricity")
    axes[0].set_ylabel("Power (kW)")
    axes[0].grid(True)
    axes[0].set_ylim((-500, 500))
    axes[0].set_yticks(np.arange(-500, 500 + 250, 250))

    ax2 = axes[1].twinx()
    (heat.total / 1000).plot(
        ax=axes[1],
        label="Total heating load",
        alpha=1,
        linewidth=linewidth,
        color=colors_array[6],
    )
    (-cool.total / 1000).plot(
        ax=axes[1],
        label="Total cooling load",
        alpha=1,
        linewidth=linewidth,
        color=colors_array[4],
    )
    weather["WeatherStation.Weather.Ta"].plot(
        ax=ax2,
        label="Ambient air temperature",
        alpha=alpha,
        linewidth=linewidth,
        color=colors_array[3],
    )
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend_loc = "upper left"
    legend = ax2.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc, ncols=3)
    for handle in legend.legend_handles:
        handle.set_linewidth(1.5)

    axes[1].set_title("Cooling and Heating")
    axes[1].set_ylabel("Power (kW)")
    axes[1].grid(True)
    axes[1].set_xlabel(None)
    axes[1].set_ylim((-250, 1250))
    ax2.set_ylabel("Temperature ($^\circ\mathrm{C}$)")

    lb = -5
    ub = 25
    n_ticks = 6
    eps = (ub - lb) / n_ticks
    # ax2.set_yticks(np.arange(lb, ub + eps, eps))
    # ax2.set_ylim((lb - 0.5 * eps, ub))
    ax2.set_ylim((-5, 25))

    for ax in axes:
        ax.grid("on", axis="x", which="minor")
        ax.grid(True)
        # ax.minorticks_on()

    fig.tight_layout()
    return fig, axes


#%%

parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
plt.style.use(parent_dir / "style.mplstyle")

colors = plt.rcParams["axes.prop_cycle"]
colors_array = colors.by_key()["color"]

colors_array = list(map(lambda x: adjust_lightness(x, 0.8), colors_array))

with open(parent_dir / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]
    dir_series_reduced = cfg["reduced_datapath"]

RESOLUTION = "15min"
PATH = Path(dir_series_reduced) / RESOLUTION
figure_path = parent_dir.joinpath("figures")

if __name__ == "__main__":
    elec = pd.read_csv(
        os.path.join(PATH, "electricity_P.csv.gz"),
        compression="gzip",
        index_col=0,
        parse_dates=True,
    )
    cool = pd.read_csv(
        os.path.join(PATH, "cooling_P.csv.gz"),
        compression="gzip",
        index_col=0,
        parse_dates=True,
    )
    heat = pd.read_csv(
        os.path.join(PATH, "heating_P.csv.gz"),
        compression="gzip",
        index_col=0,
        parse_dates=True,
    )
    weather = pd.read_csv(
        os.path.join(PATH, "weather.csv.gz"),
        compression="gzip",
        index_col=0,
        parse_dates=True,
    )

    t0 = pd.Timestamp("2021-03-01 00:00:00", tz="UTC").to_datetime64()
    t1 = pd.Timestamp("2021-03-08 00:00:00", tz="UTC").to_datetime64()

    elec_filtered = elec[elec.index.to_series().between(t0, t1)]
    cool_filtered = cool[cool.index.to_series().between(t0, t1)]
    heat_filtered = heat[heat.index.to_series().between(t0, t1)]
    weather_filtered = weather[weather.index.to_series().between(t0, t1)]

    # %%
    fig, _ = plot_timeseries(
        (elec_filtered, cool_filtered, heat_filtered, weather_filtered),
        alpha=1,
        linewidth=1,
    )
    fig.savefig(figure_path / "time_series_week.pdf", bbox_inches="tight")
    plt.show()
