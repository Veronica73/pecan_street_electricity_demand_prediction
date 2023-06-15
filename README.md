# Pecan Street Electricity Demand Forecast

## Project Overview
The purpose of the project is to give you a taste of data mining in the power system. In this project, we forecast the hourly electricity consumption of residential households to better schedule the electricity supply. Since the using pattern varies dramatically among users, clustering is first applied to group users with similar using pattern. Then, forecast model is fitted cluster-wise. In this way, we could obtain higher prediction accuracy.

# Dataset
The data for this project is individual users' daily energy consumption from Pecan Street. You could download the data [here](https://www.kaggle.com/datasets/zhitingzheng/pecan-street-electricity-data).

The selected dataset contains electricity usage of 346 residential households from 2016 Jan 1st to 2016 April 30th, with the granularity of one minute. Not all the customer have records that span the whole four months. i.e., some users have one or two months of data, or some missing values within the four months.

## EDA & Data Preprocessing
[Notebook](https://github.com/Veronica73/pecan_street_electricity_demand_prediction/blob/main/EDA%20%26%20Data%20Preprocessing.ipynb) 

Highlights from preliminary analysis:
* Converted the data into hourly data.
* On 2016-03-13 2 a.m., all user data are missing. I assumed that there was a blackout or system malfunction. I imputed the value by average of 1 a.m. and 3 a.m.
* There exists users with negative electricity consumption values, which is not possible. For users with a few/large amount of negative data, I handled them differently to ensure data integraty and accuracy. Details are in the notebook.
* Computed user's consumption pattern by averaging the hourly consumptions over days.


## Clustering
[Notebook](https://github.com/Veronica73/pecan_street_electricity_demand_prediction/blob/main/Cluster.ipynb)  

This notebook explored a variety of clustering algorithoms, ranging from partitioning methods (k-means, k-medoids), hierarchical methods (BIRCH), to density-based methods (DBSCAN). This notebook also contains some explanations of the clustering algorithms, which would be good tutorials if you are not familiar with these methods.

Before clustering, a local smoothing method is applied to the data to mitigate the "spikes" in the data. All the intuitions and algorithm are introduced in the notebook.


## Electricity Forecast
[Notebook-Time Series](https://github.com/Veronica73/pecan_street_electricity_demand_prediction/blob/main/Electricity%20Forecast%20--%20Time%20Series.ipynb)   

[Notebook-Neural Network](https://github.com/Veronica73/pecan_street_electricity_demand_prediction/blob/main/Electricity%20Forecast%20--%20Neural%20Network.ipynb) 

In this part, we test the performance of both **time series** models and **neural network** models, and the model comparison can be found [here](https://github.com/Veronica73/pecan_street_electricity_demand_prediction/blob/main/Model-Comparison.ipynb).






