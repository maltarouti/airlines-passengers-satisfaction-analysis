#Import required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# Use dark style for seaborn plots
sns.set_style("dark")

# Use PuBu palette for seaborn plots
palette = 'PuBu'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pickle


# Read the dataset using pandas library 
dataset = pd.read_csv('dataset.csv')

# Get the first 5 records
print(dataset.head())

# Display all columns and their type
print(dataset.info() )

# Get the number of missing values
print(dataset.isna().sum() )

# Get the number of duplicate rows
print(dataset.duplicated().sum() )

# Rename the Satisfaction columns
dataset.rename(columns={'satisfaction_v2':'Satisfaction'}, inplace=True)

# Fill the missing values in Arrival Delay with zero
dataset['Arrival Delay in Minutes'].fillna(0, inplace=True) 

# Convert the Flight Distance from miles to Kilos
dataset['Flight Distance'] = dataset['Flight Distance']*1.609

# Move Satisfaction column to the end of the dataset
column = dataset.pop('Satisfaction')
dataset.insert(len(dataset.columns), 'Satisfaction', column)

# Check the null values
print(dataset.isna().sum())


# Get the first 5 records after cleaning
print(dataset.head())

# Get the total number of satisfied and unsatisfied passengers
plt.figure(figsize=(11, 7))
sns.countplot(x='Satisfaction', data=dataset, palette=palette)
plt.ylabel('Count', fontsize=15, labelpad=12)
plt.xlabel('Satisfaction', fontsize=15)

# Get the satisfaction rate with respect to Gender, Type of Travel, Class, and customer type
categorical_columns = ['Gender', 'Type of Travel', 'Class', 'Customer Type']

plot, plot_map = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plot_map = plot_map.ravel()

for index, column in enumerate(categorical_columns):
    plt.subplot(2, 2, index+1)
    plot_map[index] = sns.countplot(x=column, hue='Satisfaction', data=dataset, palette=palette)


# Get the highiest relationship between columns
print(dataset.corr())

plt.figure(figsize=(20, 14))
sns.heatmap(dataset.corr(), cmap=palette, annot=True)
plt.tight_layout()


# Get all unique types of categorical columns
print("{}: {}".format('Satisfaction', dataset['Satisfaction'].unique()))
print("{}: {}".format('Gender', dataset['Gender'].unique()))
print("{}: {}".format('Customer Type', dataset['Customer Type'].unique()))
print("{}: {}".format('Type of Travel', dataset['Customer Type'].unique()))
print("{}: {}".format('Class', dataset['Customer Type'].unique()))


# Change all the value from strings to integers
dataset['Satisfaction'] = dataset['Satisfaction'].replace({'neutral or dissatisfied':0, 'satisfied':1})
dataset['Gender'] = dataset['Gender'].replace({'Male':0, 'Female':1})
dataset['Customer Type'] = dataset['Customer Type'].replace({'disloyal Customer':0, 'Loyal Customer':1})
dataset['Type of Travel'] = dataset['Type of Travel'].replace({'Personal Travel':0, 'Business travel':1})
dataset['Class'] = dataset['Class'].replace({'Eco':0, 'Business':1, 'Eco Plus':2})

print(dataset.head())

# Drop unwanted columns
dataset.drop(['id', 'Age'], axis=1, inplace=True)

# Split the dataset into training and testing sets
features = dataset.drop(['Satisfaction'], axis=1)
target = dataset['Satisfaction']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=88)

# Create the model
model =  LogisticRegression(C=10, tol=0.01, solver='lbfgs', max_iter=10000) 

# train the model
model.fit(x_train, y_train)

# Test the model 
result = model.score(x_test, y_test)
print('Score: {}%'.format(round(result*100)))


# Create a small test dataset manually
new_people = [
                [0, 1, 1, 1, 2991, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [1, 1, 1, 1, 5000, 1, 1, 2, 5, 2, 1, 3, 4, 1, 3, 3, 1, 1, 1, 2, 1]
             ]

# Convert the list to Pandas DataFrame
new_people = pd.DataFrame(data=new_people, columns=x_train.columns)

print(new_people)

# Use the machine learning model to do predictions
# predict using the machine learning model we built
results = model.predict(new_people)

# Change all ones and zeros with Satisifed and Unsatisfied 
fixed_results = []
for index, value in enumerate(results):
    if value == 1:
        fixed_results.append('Satisifed')
    else:
        fixed_results.append('Unsatisfied')


# Reshape the list to 1-column
fixed_results = np.array(fixed_results).reshape(-1, 1)

# Convert the predictions as pandas DataFrame
predictions = pd.DataFrame(fixed_results, columns=['Satisfaction'])

# Add the predictions to the small dataset we created
new_people['Satisfaction'] = predictions

print(new_people)


# Save the model
with open("passengers_satisfaction_classifier.pickle", "wb") as f:
    pickle.dump(model, f)