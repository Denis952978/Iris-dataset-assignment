# Import necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column for species
df['species'] = iris.target

# Map species integers to their names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (no missing values here, but this is where you would handle them)

# Task 2: Basic Data Analysis
print("\nSummary Statistics:")
print(df.describe())

# Group data by species and calculate mean of each feature
grouped_data = df.groupby('species').mean()
print("\nMean values for each species:")
print(grouped_data)

# Find the species with the highest average sepal length
max_sepal_length = grouped_data['sepal length (cm)'].idxmax()
print(f"\nThe species with the highest average sepal length is {max_sepal_length}.")

# Task 3: Data Visualizations
# 1. Line Chart: Mean sepal length by species
plt.figure(figsize=(10, 6))
grouped_data['sepal length (cm)'].plot(kind='line', marker='o', title="Mean Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.grid()
plt.show()

# 2. Bar Chart: Average petal width by species
plt.figure(figsize=(10, 6))
grouped_data['petal width (cm)'].plot(kind='bar', title="Average Petal Width by Species", color=['skyblue', 'orange', 'green'])
plt.xlabel("Species")
plt.ylabel("Average Petal Width (cm)")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal width (cm)'], kde=True, bins=20, color='purple')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal length vs petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='bright')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

# Observations
print("\nObservations:")
print("1. Setosa species has the smallest average sepal and petal lengths.")
print("2. Virginica species has the largest average petal width.")
print("3. Sepal length and petal length show a positive correlation across species.")
print("4. Sepal width values are approximately normally distributed.")
