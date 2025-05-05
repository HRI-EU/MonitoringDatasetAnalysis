# A Real-World Energy Management data set from a Smart Company Building for Optimization and Machine Learning

This repository contains the Python code used for validating the real world energy management smart company data set, published under https://doi.org/10.5061/dryad.73n5tb363.

To run the code, we recommend using Python 3.10 and installing the requirements via pip
```shell
pip install -r requirements.txt
```

Furthermore, `config.yaml` should be configured to contain the respective paths to the data set.

The file `issue_template.yml` outlines the format of issue files as found in the dataset.

-----

The `src` directory contains the following scripts used to create the figures and statistics for the publication accompanying the data set:
* `create_reduced_dataset.py`: Script for creating the reduced aggregated data set from the full data set.
* `downsample_measurements.py`: Script for downsampling equidistantly sampled 1min time series to 15min and 1h resolution.
* `error_statistics_p_vs_w.py`: Script for comparing P measurements vs. W measurements, yielding plots and statistics. 
* `energy_flow_sankey.py`: Script for generating the Sankey diagram illustrating overall electrical energy flows.
* `issues_statistics.py`: Script for creating statistics table on the automatically detected and manually specified issues of the data set.
* `representative_time_series_full.py`: Script for creating the time series plot showcasing the most important measurements over the full dataset measurement period.
* `representative_time_series_full.py`: Script for creasting the time showcasing a representative week from the dataset.
* `yearly_energy_statistics.py`: Script for creating the figure and tables for yearly energy consumption and production statistics.
* `ReadFiles.py`: Helper script used for creating reduced data set and the Sankey diagram. Contains a set of useful functions to load meters and elements of the data set effectively.

The `meters.yaml` in the root directory contains a list of all meters' uniform resource names (URNs) present in the dataset, grouped into categories.
This file is used by `ReadFiles.py` to associate categories with the respective list of URNs for easier data handling.
`style.mplstyle` contains the pyplot style used for creating figures.