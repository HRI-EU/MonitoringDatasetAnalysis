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
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import colorsys

SAMPLING = "1min"

parent_dir = Path(__file__).parent.parent


def adjust_lightness(color, amount=0.5):
    if color in matplotlib.colors.cnames:
        c = matplotlib.colors.cnames[color]
    if isinstance(color, str) and (color[0] == "#" or len(color) == 6):  #
        c = matplotlib.colors.ColorConverter.to_rgb(color)
    else:
        c = color

    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


with open(parent_dir / "config.yaml", "r") as file:
    cfg = yaml.safe_load(file)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]
    dir_series_reduced = cfg["reduced_datapath"]

plt.style.use(parent_dir / "style.mplstyle")
figure_path = parent_dir.joinpath("figures")

data_path = Path(dir_series)

colors = plt.rcParams["axes.prop_cycle"]
colors_array = colors.by_key()["color"]

colors_array = list(map(lambda x: adjust_lightness(x, 0.8), colors_array))

#%%
t0 = pd.Timestamp("2021-05-01 08:00:00", tz="UTC")
t1 = pd.Timestamp("2021-05-01 12:00:00", tz="UTC")

df_p_z15 = pd.read_csv(
    data_path / "H1.Z15" / f"H1.Z15.P_corrected_resampled_{SAMPLING}.csv.gz",
    index_col=0,
    parse_dates=True,
)
df_w_z15 = pd.read_csv(
    data_path / "H1.Z15" / f"H1.Z15.w_corrected_resampled_{SAMPLING}.csv.gz",
    index_col=0,
    parse_dates=True,
)

df_p_z15 = df_p_z15[df_p_z15.index.to_series().between(t0, t1)]
df_w_z15 = df_w_z15[df_w_z15.index.to_series().between(t0, t1)]

df_p_z20 = pd.read_csv(
    data_path / "H1.Z20" / f"H1.Z20.P_corrected_resampled_{SAMPLING}.csv.gz",
    index_col=0,
    parse_dates=True,
)
df_w_z20 = pd.read_csv(
    data_path / "H1.Z20" / f"H1.Z20.w_corrected_resampled_{SAMPLING}.csv.gz",
    index_col=0,
    parse_dates=True,
)

df_p_z20 = df_p_z20[df_p_z20.index.to_series().between(t0, t1)]
df_w_z20 = df_w_z20[df_w_z20.index.to_series().between(t0, t1)]

#%%
t0_plot = pd.Timestamp("2021-05-01 08:00:00", tz="UTC")
t1_plot = pd.Timestamp("2021-05-01 12:00:00", tz="UTC")

fig, axs = plt.subplots(nrows=2, height_ratios=(3, 1), dpi=300, sharex=True)

ax = axs[0]
(df_p_z15[df_p_z15.columns[0]] / 1000).plot(
    ax=ax, label="$P$", alpha=0.8, color=colors_array[0]
)
# (df_p_z17[df_p_z17.columns[0]] / 1000).plot(ax=ax, label="$P$", alpha=0.8, color=colors_array[0], drawstyle="steps-post")
dt = df_w_z15.index.diff()[1].total_seconds() / 60 / 60
dW = df_w_z15[df_w_z15.columns[0]].diff().shift(-1) / dt
dW.plot(ax=ax, label="$\Delta W/\Delta t$", alpha=0.8, color=colors_array[1])
# dW.plot(ax=ax, label="$\Delta W/\Delta t$", alpha=0.8, color=colors_array[1], drawstyle="steps-post")
ax.grid(True)
ax.set_ylabel("Power (kW)")
ax.set_xlabel(None)
ax.legend(loc="lower left")
ax.set_title("H1.Z15")
ax.set_ylim((-175, 100))
ax.xaxis.set_major_formatter("")
ax.xaxis.set_minor_formatter("")

ax = axs[1]
P = df_p_z15[df_p_z15.columns[0]] / 1000
error = P - dW
error.plot(ax=ax, color=colors_array[2])
print(error.abs().sum())
ax.set_xlabel("Time of Day")
ax.set_ylim((-60, 60))
ax.set_ylabel("$\epsilon$ (kW)")
ax.grid(True)
fig.tight_layout()
fig.savefig(figure_path / "comparison_p_vs_w_h1_z15.pdf")
plt.show()

#%%
fig, axs = plt.subplots(nrows=2, height_ratios=(3, 1), dpi=300, sharex=True)

ax = axs[0]
(df_p_z20[df_p_z20.columns[0]] / 1000).plot(
    ax=ax, label="$P$", alpha=0.8, color=colors_array[0]
)
# (df_p_z17[df_p_z17.columns[0]] / 1000).plot(ax=ax, label="$P$", alpha=0.8, color=colors_array[0], drawstyle="steps-post")
dt = df_w_z20.index.diff()[1].total_seconds() / 60 / 60
dW = df_w_z20[df_w_z20.columns[0]].diff().shift(-1) / dt
dW.plot(ax=ax, label="$\Delta W/\Delta t$", alpha=0.8, color=colors_array[1])
# dW.plot(ax=ax, label="$\Delta W/\Delta t$", alpha=0.8, color=colors_array[1], drawstyle="steps-post")
ax.grid(True)
ax.set_ylabel("Power (kW)")
ax.set_xlabel(None)
ax.legend(loc="lower left")
ax.set_title("H1.Z20")
# ax.set_ylim((25, 100))
ax.xaxis.set_major_formatter("")
ax.xaxis.set_minor_formatter("")

ax = axs[1]
P = df_p_z20[df_p_z20.columns[0]] / 1000
error = P - dW
error.plot(ax=ax, color=colors_array[2])
ax.set_xlabel("Time of Day")
ax.set_ylim((-60, 60))
ax.grid(True)
ax.set_ylabel("$\epsilon$ (kW)")
print(error.abs().sum())
fig.tight_layout()
fig.savefig(figure_path / "comparison_p_vs_w_h1_z20.pdf")
plt.show()
