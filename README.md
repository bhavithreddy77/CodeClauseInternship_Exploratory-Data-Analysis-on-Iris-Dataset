# CodeClauseInternship_Exploratory-Data-Analysis-on-Iris-Dataset
#This is my first github project.
<br>
#Author- Bhavith Reddy Anugu

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) column
df['species'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Check data types of columns
df.dtypes

# Summary statistics of the numerical features
df.describe()

# Pairplot to see relationships between features, colored by species
sns.pairplot(df, hue='species', palette='Set1')
plt.show()

# Plot histograms for each feature
df.drop('species', axis=1).hist(bins=20, figsize=(10, 8), layout=(2, 2))
plt.tight_layout()
plt.show()

# Boxplot for Sepal Length across species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=df, palette='Set1')
plt.title("Sepal Length by Species")
plt.show()

# Boxplot for Petal Length across species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df, palette='Set1')
plt.title("Petal Length by Species")
plt.show()

# Calculate the correlation matrix
corr_matrix = df.drop('species', axis=1).corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairwise density contours
sns.kdeplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', fill=True, common_norm=False, palette='Set1')
plt.title("Pairwise Density Contours: Sepal Length and Petal Length")
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df['sepal length (cm)'], df['petal length (cm)'], df['petal width (cm)'], c=df['species'].map({'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}))

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')

plt.title("3D Scatter Plot of Sepal Length, Petal Length, and Petal Width")
plt.show()

