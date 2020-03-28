# 1. Introduction

Time series forecasting is an important area of machine learning that is often neglected.

- It is important because **there are so many prediction problems that involve a time component.** 
- These problems are neglected because it is this time component that makes time series problems more difficult to handle.



**Time Series:**

A normal machine learning dataset is a collection of observations. For example:

```python
observation #1
observation #2
observation #3
```

Time does play a role in normal machine learning datasets:

- Predictions are made for new data when the actual outcome may not be known until some future date. The future is being predicted, but all prior observations are almost always treated equally. 
- Perhaps with some very minor temporal dynamics to overcome the idea of “*concept drift*” such as only using the last year of observations rather than all data available.

A time series dataset is different:

- **Time series adds an explicit order dependence between observations: a time dimension.** For example:

  ```python
  Time #1, observation
  Time #2, observation
  Time #3, observation
  ```

- This additional dimension is both a constraint and a structure that provides a source of additional information.

> A time series is a sequence of observations taken sequentially in time.
>
> *-- Page 1, [Time Series Analysis: Forecasting and Control](http://www.amazon.com/dp/1118675029?tag=inspiredalgor-20).*



**Describing vs. Predicting**

We have different goals depending on whether we are interested in understanding a dataset or making predictions:

- Understanding a dataset, called *time series analysis*, can help to make better predictions, but is not required and can result in a large technical investment in time and expertise not directly aligned with the desired outcome, which is forecasting the future.

> In descriptive modeling, or time series analysis, a time series is modeled to determine its components in terms of seasonal patterns, trends, relation to external factors, and the like. … 
>
> In contrast, time series forecasting uses the information in a time series (perhaps with additional information) to forecast future values of that series
>
> *-- Page 18-19, [Practical Time Series Forecasting with R: A Hands-On Guide](http://www.amazon.com/dp/0997847913?tag=inspiredalgor-20).*



**Time Series Analysis**

When using classical statistics, the primary concern is the analysis of time series.

- Time series analysis involves **developing models that best capture or describe an observed time series in order to understand the underlying causes.** This field of study seeks the “*why*” behind a time series dataset.

- This often involves making assumptions about the form of the data and decomposing the time series into constitution components.

- The quality of a descriptive model is determined by how well it describes all available data and the interpretation it provides to better inform the problem domain.

> The primary objective of time series analysis is to develop mathematical models that provide plausible descriptions from sample data
>
> — Page 11, [Time Series Analysis and Its Applications: With R Examples](http://www.amazon.com/dp/144197864X?tag=inspiredalgor-20)



**Time Series Forecasting**

- Forecasting involves taking models fit on historical data and using them to predict future observations.

- Descriptive models can borrow for the future (i.e. to smooth or remove noise), they only seek to best describe the data.

- An important distinction in forecasting is that the **future is completely unavailable** and must only be estimated from what has already happened. 

> The purpose of time series forecasting is generally twofold: to understand or model the stochastic mechanisms that gives rise to an observed series and to predict or forecast the future values of a series based on the history of that series
>
> — Page 1, [Time Series Analysis: With Applications in R](http://www.amazon.com/dp/0387759581?tag=inspiredalgor-20).



**Components of Time Series:**

Time series analysis provides a body of techniques to better understand a dataset.

Perhaps the most useful of these is the decomposition of a time series into 4 constituent parts:

1. **Level**. The baseline value for the series if it were a straight line.
2. **Trend**. The optional and often linear increasing or decreasing behavior of the series over time.
3. **Seasonality**. The optional repeating patterns or cycles of behavior over time.
4. **Noise**. The optional variability in the observations that cannot be explained by the model.

All time series have a level, most have noise, and the trend and seasonality are optional.

These constituent components can be thought to combine in some way to provide the observed time series. For example, they may be added together to form a model as follows:

```
y = level + trend + seasonality + noise
```

Assumptions can be made about these components both in behavior and in how they are combined, which allows them to be modeled using traditional statistical methods.



**Concerns of Forecasting：**

Use the Socratic method and ask lots of questions to help zoom in on the specifics of your [predictive modeling problem](https://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/). For example:

1. **How much data do you have available and are you able to gather it all together?** More data is often more helpful, offering greater opportunity for exploratory data analysis, model testing and tuning, and model fidelity.
2. **What is the time horizon of predictions that is required? Short, medium or long term?** Shorter time horizons are often easier to predict with higher confidence.
3. **Can forecasts be updated frequently over time or must they be made once and remain static?**
   - Updating forecasts as new information becomes available often results in more accurate predictions.
4. **At what temporal frequency are forecasts required?** Often forecasts can be made at a lower or higher frequencies, allowing you to harness down-sampling, and up-sampling of data, which in turn can offer benefits while modeling.



Time series data often requires **cleaning, scaling, and even transformation**. For example:

- **Frequency**. Perhaps data is provided at a frequency that is too high to model or is unevenly spaced through time requiring resampling for use in some models.
- **Outliers**. Perhaps there are corrupt (dirty data) or extreme outlier values that need to be identified and handled.
- **Missing**. Perhaps there are gaps or missing data that need to be interpolated or imputed.

Often time series problems are **real-time**, continually providing new opportunities for prediction. This adds an honesty to time series forecasting that **quickly flushes out bad assumptions**, errors in modeling and all the other ways that we may be able to fool ourselves.



**Examples of Time Series Forecasting:**

- Forecasting the corn(玉米) yield in tons by state each year.
- Forecasting whether an EEG(脑电图) trace in seconds indicates a patient is having a seizure or not.
- Forecasting the closing price of a stock each day.
- Forecasting the birth rate at all hospitals in a city each year.
- Forecasting product sales in units sold each day for a store.
- Forecasting the number of passengers through a train station each day.
- Forecasting unemployment(失业率) for a state each quarter(季度).
- Forecasting utilization demand on a server(服务器) each hour.
- Forecasting the size of the rabbit population in a state each breeding season(繁殖季节).
- Forecasting the average price of gasoline in a city each day.

**Summary**

In this post, you discovered time series forecasting. Specifically, you learned:

- About time series data and the difference between time series analysis and time series forecasting.
- The constituent components that a time series may be decomposed into when performing an analysis.
- Examples of time series forecasting problems to make these ideas concrete.





# 2. Datasets

## Univariate Time Series Datasets

Time series datasets that only have one variable are called univariate datasets.

These datasets are a great place to get started because:

- They are so simple and easy to understand.
- You can plot them easily in excel or your favorite plotting tool.
- You can easily plot the predictions compared to the expected results.
- You can quickly try and evaluate a suite of traditional and newer methods.

There are many sources of time series dataset, such as the “[*Time Series Data Library*](https://robjhyndman.com/tsdl/)” created by [Rob Hyndman](http://robjhyndman.com/), Professor of Statistics at Monash University, Australia

### Shampoo Sales Dataset

This dataset describes the monthly number of sales of shampoo over a 3 year period. The units are a sales count and there are 36 observations.  Below is a sample of the first 5 rows of data including the header row.

```
"Month","Sales of shampoo over a three year period"
"1-01",266.0
"1-02",145.9
"1-03",183.1
"1-04",119.3
"1-05",180.3
```

Below is a plot of the entire dataset.

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/11/Shampoo-Sales-Dataset.png width=300 />

The dataset shows an increasing trend and possibly some seasonal component.

- [Download the dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv).

### Minimum Daily Temperatures Dataset

This dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia. The units are in degrees Celsius and there are 3650 observations. The source of the data is credited as the Australian Bureau of Meteorology.

Below is a sample of the first 5 rows of data including the header row.

```
"Date","Daily minimum temperatures in Melbourne, Australia, 1981-1990"
"1981-01-01",20.7
"1981-01-02",17.9
"1981-01-03",18.8
"1981-01-04",14.6
"1981-01-05",15.8
```

Below is a plot of the entire dataset.

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/11/Minimum-Daily-Temperatures.png width =400 />

The dataset shows a strong seasonality component and has a nice fine grained detail to work with.

- [Download the dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv).

### Monthly Sunspot Dataset

This dataset describes a monthly count of the number of observed sunspots for just over 230 years (1749-1983). The units are a count and there are 2,820 observations. The source of the dataset is credited to Andrews & Herzberg (1985).

Below is a sample of the first 5 rows of data including the header row.

```
"Month","Zuerich monthly sunspot numbers 1749-1983"
"1749-01",58.0
"1749-02",62.6
"1749-03",70.0
"1749-04",55.7
"1749-05",85.0
```

Below is a plot of the entire dataset.

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/11/Daily-Female-Births-Dataset.png width = 400 />

The dataset shows seasonality with large differences between seasons.

- [Download the dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv).

### Daily Female Births Dataset

This dataset describes the number of daily female births in California in 1959. The units are a count and there are 365 observations. The source of the dataset is credited to Newton (1988).

Below is a sample of the first 5 rows of data including the header row.

```
"Date","Daily total female births in California, 1959"
"1959-01-01",35
"1959-01-02",32
"1959-01-03",30
"1959-01-04",31
"1959-01-05",44
```

Below is a plot of the entire dataset.

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/11/Daily-Female-Births-Dataset.png width=400 />

Daily Female Births Dataset

- [Download the dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv).

## Multivariate Time Series Datasets

Multivariate datasets are generally more challenging and are the sweet spot for machine learning methods.

A great source of multivariate time series data is the **[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).** At the time of writing, there are 63 time series datasets that you can download for free and work with.

### EEG Eye State Dataset

This dataset describes EEG data for an individual and whether their eyes were open or closed.

- The objective of the problem is to predict whether eyes are open or closed given EEG data alone.
- This is a classification predictive modeling problems and there are a total of 14,980 observations and 15 input variables. 
- The class value of ‘1’ indicates the eye-closed and ‘0’ the eye-open state. 
- Data is ordered by time and observations were recorded over a period of 117 seconds.

Below is a sample of the first 5 rows with no header row.

```
4329.23,4009.23,4289.23,4148.21,4350.26,4586.15,4096.92,4641.03,4222.05,4238.46,4211.28,4280.51,4635.9,4393.85,0
4324.62,4004.62,4293.85,4148.72,4342.05,4586.67,4097.44,4638.97,4210.77,4226.67,4207.69,4279.49,4632.82,4384.1,0
4327.69,4006.67,4295.38,4156.41,4336.92,4583.59,4096.92,4630.26,4207.69,4222.05,4206.67,4282.05,4628.72,4389.23,0
4328.72,4011.79,4296.41,4155.9,4343.59,4582.56,4097.44,4630.77,4217.44,4235.38,4210.77,4287.69,4632.31,4396.41,0
4326.15,4011.79,4292.31,4151.28,4347.69,4586.67,4095.9,4627.69,4210.77,4244.1,4212.82,4288.21,4632.82,4398.46,0
```



- [Learn More](http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)

### Occupancy Detection Dataset

This dataset describes measurements of a room and the objective is to predict whether or not the room is occupied.

There are 20,560 one-minute observations taken over the period of a few weeks. This is a classification prediction problem. There are 7 attributes including various light and climate properties of the room.

The source for the data is credited to Luis Candanedo from UMONS.

Below is a sample of the first 5 rows of data including the header row.

```
"date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"
"1","2015-02-04 17:51:00",23.18,27.272,426,721.25,0.00479298817650529,1
"2","2015-02-04 17:51:59",23.15,27.2675,429.5,714,0.00478344094931065,1
"3","2015-02-04 17:53:00",23.15,27.245,426,713.5,0.00477946352442199,1
"4","2015-02-04 17:54:00",23.15,27.2,426,708.25,0.00477150882608175,1
"5","2015-02-04 17:55:00",23.1,27.2,426,704.5,0.00475699293331518,1
"6","2015-02-04 17:55:59",23.1,27.2,419,701,0.00475699293331518,1
```

The data is provided in 3 files that suggest the splits that may be used for training and testing a model.

- [Learn More](http://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

### Ozone Level Detection Dataset

This dataset describes 6 years of ground ozone concentration (地面臭氧浓度) observations and the objective is to predict whether it is an “ozone day” or not.

The dataset contains 2,536 observations and 73 attributes. This is a classification prediction problem and the final attribute indicates the class value as “1” for an ozone day and “0” for a normal day.

Two versions of the data are provided, eight-hour peak set and one-hour peak set. I would suggest using the one hour peak set for now.

- [Learn More](http://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection)





# Time Series Forecasting as Supervised Learning

Time series forecasting can be framed as a supervised learning problem.

This re-framing of your time series data allows you access to the suite of standard linear and nonlinear machine learning algorithms on your problem.

In this post, you will discover how you can re-frame your time series problem as a supervised learning problem for machine learning. After reading this post, you will know:

- What supervised learning is and how it is the foundation for all [predictive modeling](https://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/) machine learning algorithms.
- The sliding window method for framing a time series dataset and how to use it.
- How to use the sliding window for multivariate data and multi-step forecasting.



## Supervised Machine Learning

The majority of practical machine learning uses supervised learning.

Supervised learning is where you have input variables (**X**) and an output variable (**y**) and you use an algorithm to learn the mapping function from the input to the output.
$$
Y = f(X)
$$
The goal is to approximate the real underlying mapping so well that when you have new input data (**X**), you can predict the output variables (**y**) for that data.

Below is a contrived (捏造的) example of a supervised learning dataset where each row is an observation comprised of one input variable (**X**) and one output variable to be predicted (**y**).

```
X, y
5, 0.9
4, 0.8
5, 1.0
3, 0.7
4, 0.9
```

It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.

- We know the correct answers; 
- the algorithm iteratively makes predictions on the training data and is corrected by making updates. 
- Learning stops when the algorithm achieves an acceptable level of performance.

Supervised learning problems can be further grouped into regression and classification problems.

- **Classification**: A classification problem is when the output variable is a category, such as “*red*” and “*blue*” or “*disease*” and “*no disease*.”
- **Regression**: A regression problem is when the output variable is a real value, such as “*dollars*” or “*weight*.” The contrived example above is a regression problem.



## Sliding Window For Time Series Data

Time series data can be phrased as supervised learning. Given a sequence of numbers for a time series dataset, we can restructure the data to look like a supervised learning problem. 

- We can do this by using previous time steps as input variables and use the next time step as the output variable.

Let’s make this concrete with an example. Imagine we have a time series as follows:

```
time, measure
1, 100
2, 110
3, 108
4, 115
5, 120
```

We can **restructure this time series dataset** as a supervised learning problem by using the value at the previous time step to predict the value at the next time-step. Re-organizing the time series dataset this way, the data would look as follows:

```
X, y
?, 100
100, 110
110, 108
108, 115
115, 120
120, ?
```



## Sliding Window With Multivariate Time Series Data

Below is another worked example to make the sliding window method concrete for multivariate time series.

Assume we have the contrived multivariate time series dataset below with two observations at each time step. Let’s also assume that we are only concerned with predicting **measure2**.

```
time, measure1, measure2
1, 0.2, 88
2, 0.5, 89
3, 0.7, 87
4, 0.4, 88
5, 1.0, 90
```

We can re-frame this time series dataset as a supervised learning problem with **a window width of one**.

This means that we will use the previous time step values of **measure1** and **measure2**. We will also have available the next time step value for **measure1**. We will then predict the next time step value of **measure2**.

This will give us 3 input features and one output value to predict for each training pattern.

```
X1, X2, X3, y
?, ?, 0.2 , 88
0.2, 88, 0.5, 89
0.5, 89, 0.7, 87
0.7, 87, 0.4, 88
0.4, 88, 1.0, 90
1.0, 90, ?, ?
```

This example raises the question of what if we wanted to predict both **measure1** and **measure2** for the next time step?

The sliding window approach can also be used in this case.

Using the same time series dataset above, we can phrase it as a supervised learning problem where we predict both **measure1** and **measure2** with the same window width of one, as follows.

```
X1, X2, y1, y2
?, ?, 0.2, 88
0.2, 88, 0.5, 89
0.5, 89, 0.7, 87
0.7, 87, 0.4, 88
0.4, 88, 1.0, 90
1.0, 90, ?, ?
```

Not many supervised learning methods can handle the prediction of multiple output values without modification, but some methods, like artificial neural networks, have little trouble.

**We can think of predicting more than one value as predicting a sequence**. In this case, we were predicting two different output variables, but we may want to **predict multiple time-steps ahead of one output variable.**This is called multi-step forecasting and is covered in the next section.

## Sliding Window With Multi-Step Forecasting

The number of time steps ahead to be forecasted is important.

Again, it is traditional to use different names for the problem depending on the number of time-steps to forecast:

- **One-Step Forecast**: This is where the next time step (t+1) is predicted.
- **Multi-Step Forecast**: This is where two or more future time steps are to be predicted.

All of the examples we have looked at so far have been one-step forecasts.

There are are a number of ways to model multi-step forecasting as a supervised learning problem. We will cover some of these alternate ways in a future post. For now, we are focusing on framing multi-step forecast using the sliding window method.

Consider the same univariate time series dataset from the first sliding window example above:

```
time, measure
1, 100
2, 110
3, 108
4, 115
5, 120
```

We can frame this time series as **a two-step forecasting** dataset for supervised learning with a **window width of one**, as follows:

```
X1, y1, y2
? 100, 110
100, 110, 108
110, 108, 115
108, 115, 120
115, 120, ?
120, ?, ?
```

We can see that the first row and the last two rows cannot be used to train a supervised model.

It is also a good example to show the burden on the input variables. Specifically, that a supervised model only has **X1** to work with in order to predict both **y1** and **y2**.

*Careful thought and experimentation are needed on your problem to **find a window width** that results in acceptable model performance.*



# How to Load and Explore Time Series Data in Python

<https://machinelearningmastery.com/category/time-series/page/5/>

