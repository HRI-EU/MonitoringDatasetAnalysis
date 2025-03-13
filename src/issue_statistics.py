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
from pathlib import Path

# Directory containing the yaml files
# -> We assume that the data has been copied to a directory 2 levels upward
parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

with open(parent_dir / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    issues_automatic_dir = cfg["issues_automatic_path"]
    issues_manual_dir = cfg["issues_manual_path"]
    dir_series = cfg["datapath"]

meters_to_ignore = [
    "1",
    "H3.Z322",
    "V.Z323",
    "V.Z120",
    "V.Z121",
    "V.Z122",
    "V.Z123",
    "V.Z124",
    "V.Z85",
    "V.Z86",
    "V.Z87",
    "V.Z88",
    "V.Z81c",
    "V.Z82c",
    "V.Z83",
    "V.Z323",
    "V.Z321",
    "V.Z320",
    "V.K22",
    "calc",
]

reasons_to_ignore = ["transformer_factor", "errors"]

# List to store the data for the DataFrame
data = []

issue_files = list(Path(issues_automatic_dir).glob("*.yaml")) + list(
    Path(issues_manual_dir).glob("*.yaml")
)

# Loop through all files in the directory
for filename in issue_files:
    # Check if any of the meter names in 'meters_to_ignore' is in the filename

    urn = filename.name.split("_")[0]

    if urn in meters_to_ignore:
        print(f"Skipping file: {filename} (ignored meter)")
        continue  # Skip this file if it contains an ignored meter

    print(f"Processing file: {filename}")

    if filename.name.endswith(".yaml"):
        # filepath = os.path.join(issues_dir, filename)

        # Open and load the yaml file
        with open(filename, "r") as file:
            content = yaml.safe_load(file)

            # Loop through each entry in the yaml file
            for key, entry in content.items():
                meter_name = key.split("@")[0]
                reason = entry.get("reason", None)
                time_start = entry.get("time_start", None)
                time_end = entry.get("time_end", None)

                if reason in reasons_to_ignore:
                    continue

                if isinstance(time_start, str):
                    time_start = pd.to_datetime(time_start).timestamp()

                if isinstance(time_end, str):
                    time_end = pd.to_datetime(time_end).timestamp()

                # Append the extracted data to the list
                data.append(
                    {
                        "meter_name": meter_name,
                        "reason": reason,
                        "time_start": time_start,
                        "time_end": time_end,
                    }
                )

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

#%% calculate duration
df["duration"] = df["time_end"] - df["time_start"]

# for the "zero" gaps, assume that duration was 1 min (i.e. 60 seconds)
df.loc[df["duration"] == 0, "duration"] = 60

#%% Calculate statistics

# 1) count # of meters
with open(parent_dir / "meters.yaml", "r") as f:
    meters_dict = yaml.safe_load(f)

meters = []
for category, meter_list in meters_dict.items():
    for meter in meter_list:
        if meter not in meters:
            meters.append(meter)

# Count the number of directories
num_meters = len(meters)

# substract 2, because of office trafos H2.Z35 / 36 have been replaced by H2.Z351 H2.Z361
num_meters -= 2

total_recorded_time = num_meters * 6 * 365 * 24 * 60 * 60

#%%
grouped_df = (
    df.groupby("reason")
    .agg(
        total_issues=("reason", "size"),
        average_duration=("duration", "mean"),
        total_duration=("duration", "sum"),
    )
    .reset_index()
)

# Add the new column: total time divided by total_recorded_time
grouped_df["total_time_ratio"] = grouped_df["total_duration"] / total_recorded_time

# Rename 'reason' to 'category'
grouped_df.rename(columns={"reason": "category"}, inplace=True)

print(grouped_df)

#%% make latex friendly
df_latex = grouped_df.rename(
    columns={
        "category": "Category",
        "total_issues": "Total Number of Issues",
        "average_duration": "Average Duration (in s)",
        "total_time_ratio": "Total time ratio (in $\%$)",
    }
)

df_latex = df_latex.drop(columns=["total_duration"])

df_latex["Total time ratio (in $\%$)"] = df_latex["Total time ratio (in $\%$)"] * 100
df_latex["Category"] = df_latex["Category"].replace(
    {"lasting_leap": "lasting leap", "single_leap": "single leap"}
)

df_latex.set_index("Category", inplace=True)

df_latex.loc["lasting leap", "Average Duration (in s)"] = "-"
df_latex.loc["lasting leap", "Total time ratio (in $\%$)"] = "-"
df_latex.loc["single leap", "Average Duration (in s)"] = "-"
df_latex.loc["single leap", "Total time ratio (in $\%$)"] = "-"


latex_code = df_latex.to_latex(index=True, float_format="%.2f")
print(latex_code)


#%%
longest_issues = df.sort_values(by="duration").tail(100)
longest_issues["time_start"] = pd.to_datetime(longest_issues["time_start"], unit="s")
longest_issues["time_end"] = pd.to_datetime(longest_issues["time_end"], unit="s")
