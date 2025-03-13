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

import yaml
import pandas as pd
from pathlib import Path


def get_measurement_name(file: Path, urn: str):
    return file.name.replace(urn, "").replace(file_search_string, "")[1:]


energy_measurements = ["W", "W1", "W2", "W3", "W_in", "W_out", "WQ_in", "WQ_out", "WQ"]
cumulative_measurement = energy_measurements + ["V"]

ignore_urns = []

parent_dir = Path(__file__).parent.parent

with open(parent_dir / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]

data_path = Path(dir_series)

print(f"downsampling data in {data_path}")
input("ok?")

#%%
target_resolutions = ["15min", "1h"]
file_search_string = "_corrected_resampled_1min.csv.gz"

meters = list(data_path.glob("*/"))
filtered_meters = list(filter(lambda x: x.name not in ignore_urns, meters))

n_meters = len(filtered_meters)

for i, subdirectory in enumerate(sorted(filtered_meters)):
    print(f"- processing {subdirectory.name} ({i+1}/{n_meters})")
    urn = subdirectory.name
    measurement_files = subdirectory.glob(f"*{file_search_string}")
    sorted_measurement_files = sorted(
        measurement_files, key=lambda x: get_measurement_name(x, urn)
    )
    for measurement_1min in sorted_measurement_files:
        measurement_name = get_measurement_name(measurement_1min, urn)

        print(f"\t-{measurement_name}")

        df = pd.read_csv(measurement_1min, index_col=0, parse_dates=True)

        for target_resolution in target_resolutions:
            resampler = df.resample(target_resolution)
            if measurement_name in cumulative_measurement:
                df_resampled = resampler.interpolate(method="linear")
            else:
                df_resampled = resampler.mean()

            df_resampled.to_csv(
                data_path
                / urn
                / f"{urn}.{measurement_name}_corrected_resampled_{target_resolution}.csv.gz"
            )

print("done")
