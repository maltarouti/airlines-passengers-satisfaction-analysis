# Airlines passengers satisfaction analysis

## Project Motivation
The airline is a business that offers air transportation services for traveling passengers and goods. Traveling by air is a common and efficient way to travel from a location to another. Companies follow a set of rules to ensure that air travel is safe and secure for people.

In order for an airline to be successful, It must pay close attention to some factors such as seat comfort, WiFi service, Inflight entertainment, Online support, and many more in order to maintain a high level of client satisfaction.

By analyzing one of the US airlines, we will attempt to find the best factors that contribute to high customer satisfaction ratings. We will also develop a machine learning classifier to do predictions if passengers are happy or dissatisfied with the trip by the end of this article.

## Installation process 
In order to be able to run this project, you must have [Python3](https://www.python.org/) Installed on your machine. 

You also have to install the following libraries by using pip:
  1. [Pandas](https://pypi.org/project/pandas/)
  2. [Numpy](https://pypi.org/project/numpy/)
  3. [matplotlib](https://pypi.org/project/matplotlib/)
  4. [seaborn](https://pypi.org/project/seaborn/)
  5. [scikit-learn](https://pypi.org/project/scikit-learn/)

## Files 
* airline_passenger_satisfaction.ipynb: A Jupyter notebook file contains everything such as the objectives, reading and understanding the dataset, data analysis with visualization, and modeling.
* airline_passenger_satisfaction.py: A Python file contains all the project code
* dataset.csv: The dataset we used for this project as CSV file
* model.pickle: The machine learning model as pickle file

## Objectives
The objectives of this project is to know the best factors that contribute to high customer satisfaction ratings by answering the following:
  1. What is the total number of satisfied and unsatisfied passengers in the dataset we obtained from Kaggle?
  2. What is the satisfaction rate with respect to Gender, Type of Travel, Class, and customer type?
  3. Is there a high relationship between any two factors?

We also are going to build a machine learning classifier to predict customers' satisfaction by using the Scikit-learn Python library.


## Summary 
we can say that caring about customer satisfaction is important for the business, and there are many factors to increase the level of satisfaction such as the class or the travel type. A machine learning model was built using Logistic Regression algorithm from Scikit-learn and acheived 83% accuracy 

## Acknowledgements
I would like to express my appreciation to Misk Academy and Udacity for the amazing work on the data science course and the support they give us to build this project

# Refrences
* The US airline [dataset](https://www.kaggle.com/johndddddd/customer-satisfaction)
* Take a look on the medium [post](https://medium.com/@murtada.altarouti/analyze-and-build-a-classifier-based-on-airlines-passengers-satisfaction-d98efcc5932)  
