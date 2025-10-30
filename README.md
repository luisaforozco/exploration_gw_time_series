# Ground water data investigation

This repo compiled different notebooks and files used to explore data access and methodologies to characterize time series of ground water levels.

## Requirements
The python environemnt is cluttered because of the different packages tested and used.
I'll try to update the exported env (`envs/ts_wsl.yml`) that I am using and in which I am incrementally adding dependencies, so that it can be tested in other machines. To create an env based on such a config one can run:
```bash
conda env create -f env/ts_wsl.yml
```

## Repo map
- In `src` there is code that is used in multiple places.
- `notebooks` contains jupyter notebooks that have been developed while exploring and testing data, method and functionality of the code in `src`.
### _Data notebooks_
- [bro_rest_api](notebooks/data_exploration/bro_rest_api.ipynb): how to use [BRO Rest API](https://publiek.broservices.nl/gm/gld/v1/swagger-ui/#/default/seriesAsCSV) to query GLD and how does the received data looks like.
- [dino_query](notebooks/data_exploration/dino_query.ipynb): how to query DINO DBA plus some analysis on migrated records DINO -> BRO.
- [data_exploration](notebooks/data_exploration/data_exploration.ipynb): how to get data from BRO and DINO via public ways, how does the data look like. How to import and exportbinary data generated in other notebooks and stored in folder `data`.
- [generate_test_dataset](notebooks/data_exploration/generate_test_dataset.ipynb): routine to generate a ground-truth dataset fropm migrated records.
### _Methods notebooks_ 
- [methods_exploration](notebooks/methods/methods_exploration.ipynb): Overview of methodologies fro time-series
- [wavelets](notebooks/methods/wavelets.ipynb): deep-dive into the wavelet method.
- [DTW_test](notebooks/methods/DTW_test.ipynb): Characterization of test dataset using DTW.
