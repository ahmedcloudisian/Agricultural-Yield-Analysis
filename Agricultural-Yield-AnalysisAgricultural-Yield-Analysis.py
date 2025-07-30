Agricultural Yield Analysis 

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create Sample Dataset
print("Step 1: Creating Sample Dataset...")
data = {
    'Farm_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Crop_Type': ['Wheat', 'Corn', 'Wheat', 'Corn', 'Wheat', 'Corn', 'Wheat', 'Corn', 'Wheat', 'Corn'],
    'Soil_Quality': [5, 7, 6, 8, 5, 7, 6, 8, 5, 7],
    'Water_Availability': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
    'Fertilizer_Usage': [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
    'Yield': [10, 15, 12, 16, 11, 14, 13, 17, 12, 15]
}

# Create DataFrame
df = pd.DataFrame(data)
print("\nSample Dataset:")
print(df)

# Step 2: Data Analysis
print("\nStep 2: Performing Data Analysis...")

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Correlation matrix
print("\nCalculating Correlation Matrix...")
corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Yield distribution by Crop Type
print("\nPlotting Yield Distribution by Crop Type...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Crop_Type', y='Yield', data=df)
plt.title('Yield Distribution by Crop Type')
plt.show()

# Yield vs Soil Quality
print("\nPlotting Yield vs Soil Quality...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Soil_Quality', y='Yield', hue='Crop_Type', data=df)
plt.title('Yield vs Soil Quality')
plt.show()

# Step 3: Predictive Modeling
print("\nStep 3: Performing Predictive Modeling...")

# Prepare data for modeling
X = df[['Soil_Quality', 'Water_Availability', 'Fertilizer_Usage']]
y = df['Yield']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Plot actual vs predicted
print("\nPlotting Actual vs Predicted Yield...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.show()

# Step 4: Save the Dataset
print("\nStep 4: Saving Dataset...")
df.to_csv('agricultural_yield_data.csv', index=False)
print("Dataset saved as 'agricultural_yield_data.csv'.")

print("\nProgram completed successfully!")

Data set 

You can use the dataset to perform the following analyses:
Correlation Analysis: Check how different factors (e.g., Soil_Quality, Rainfall) correlate with Yield.
Crop Comparison: Compare yields for different crop types (e.g., Wheat vs Corn).
Predictive Modeling: Train a machine learning model to predict Yield based on other features.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('complete_agricultural_yield_data.csv')

# Display the dataset
print("Dataset Loaded:")
print(df)

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Yield distribution by Crop Type
plt.figure(figsize=(8, 6))
sns.boxplot(x='Crop_Type', y='Yield', data=df)
plt.title('Yield Distribution by Crop Type')
plt.show()

# Yield vs Rainfall
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Rainfall', y='Yield', hue='Crop_Type', data=df)
plt.title('Yield vs Rainfall')
plt.show()


