# Evaluating Short-Term Forecasting of Multiple Time Series in IoT Environments

This repository holds the tools developed for the performance evaluation of several statistical, machine learning and neural network-based models for short-term forecasting of multiple time series at the IoT edge, under a unified experimental protocol.

More details can be found in the original paper:
[C. Tzagkarakis, P. Charalampidis, S. Roubakis, A. Fragkiadakis, and S. Ioannidis, “Evaluating Short-Term Forecasting of Multiple Time Series in IoT Environments,” 2022](https://arxiv.org/abs/2206.07784)

## Requirements
* Python 3.8+
* Python libraries:
    * holidays [0.14.2]
    * numpy [1.22.4]
    * openpyxl [3.0.10]
    * pandas [1.3.5]
    * requests [2.27.1]
    * scikit_learn [1.0.2]
    * scipy [1.7.3]
    * tabulate [0.8.9]
    * tensorflow [2.9.1]
    * tensorly [0.7.0]
    * tqdm [4.64.0]
    * tscv [0.1.2]
    * [DeepESN](https://github.com/lucapedrelli/DeepESN.git)
    * [BHT-ARIMA](https://github.com/huawei-noah/BHT-ARIMA)

## Download & Install

~~~~
$ git clone https://github.com/pcharala/multiple-timeseries-forecasting.git
$ cd multiple-timeseries-forecasting
$ pip install -r requirements.txt
~~~~


## Usage

##### 1. Download and pre-process the original datasets:

First, you need to run the script `download_and_preprocess.py` for downloading and pre-processing the original datasets:

~~~~
$ python3 download_and_preprocess.py
~~~~

The script considers the following optional arguments:
* '-d', '--download-path': The path for downloading the original datasets. (Default value: original-data)
* '-p', '--preproc-path': The path where the pre-processed datasets are stored. (Default value: preprocessed-data)
* '-n', '--dataset-names': The names of the datasets to download and pre-process. If not provided, all available datasets are considered. (Valid values: GuangzhouTraffic, SanFranciscoTraffic, LondonSmartMeters, EnergyConsumptionFraunhofer, ElectricityLoadDiagrams)
* '-c', '--config-path': The path of configuration files. (Default value: conf)
* '--del-original': if True, delete original datasets after pre-processing. (Default value: False)

For more information, please use the help argument:

~~~~
$ python3 download_and_preprocess.py --help
~~~~

##### 2. Execute the experiments:
Then, you may execute the experimental protocol for evaluating the performance of the considered models and datasets, by running the `run_experiments.py` script:
~~~~
$ python3 run_experiments.py
~~~~

The script considers the following optional arguments:
* '-i', '--input-path': The path where the pre-processed datasets are stored. (Default value: preprocessed-data)
* '-o', '--output-path': The path where the experimental results are stored. (Default value: results)
* '-n', '--dataset-names': The names of the datasets used in the experiments. If not provided, all available datasets are considered. (Valid values: GuangzhouTraffic, SanFranciscoTraffic, LondonSmartMeters, EnergyConsumptionFraunhofer, ElectricityLoadDiagrams)
* '-m', '--model-names': The name of the models evaluated in the experiments. If not provided, all available models are considered. (Valid values: BHTArimaModel, RFModel, SVRModel, LSTMModel, BiLSTMModel, CNNModel, DeepESNModel)
* '-c', '--config-path': The path of configuration files. (Default value: conf)
* '-r', '--mc-runs': The number of Monte Carlo runs per experimental window per dataset. (Default value: 15)
* '-e', '--error-metrics': The error metrics used for the performance evaluation. (Default value: [maape, smape, mase])
* '--cv-error-metric': The error metrics used during cross-validation (Default value: smape)
* '-s', '--summary-stats': The summary statistics to be reported, computed over Monte Carlo runs. (Default value: [mean, median, std])
* '--enable-gpu': Use GPU for NN training. (Default value: false)

For more information, please use the help argument:

~~~~
$ python3 run_experiments.py --help
~~~~


## How to reference
If you find any of this tools useful, please give credit in your publications where it is due.
