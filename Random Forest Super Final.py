#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.exceptions import ConvergenceWarning 
import warnings 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.optimizers import schedules
from tabulate import tabulate
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


data = pd.read_csv("cleaned_brfss.csv")


# ## Exploratory Data Analysis (EDA) and Feature Selection Using Chi2 and Logistic Regression

# In[4]:


data.describe()


# In[5]:


# Set display options for better presentation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Adjust width for better readability

# Print data information and missing value table
print("Data Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull())


# In[6]:


# Adjust subplots to accommodate 11 variables
nrows = 4  # Calculate number of rows needed for 11 variables
ncols = 3  # 3 columns for subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))  # Adjust figsize as needed

# Iterate over features (using enumerate for index)
for idx, feature in enumerate(data.columns):
  # Access axes based on row and column (no need to flatten for 11 variables)
  ax = axes[idx // ncols, idx % ncols]

  # Countplot with Seaborn
  sns.countplot(x=feature, data=data, ax=ax)

  # Customize the plot (optional)
  ax.set_title(f"Distribution of {feature} (Countplot)")
  ax.set_xlabel(feature)
  ax.set_ylabel("Count")
  ax.tick_params(bottom=False)  # Optional: Remove x-axis tick labels for subplots (except bottom row)

# Print value counts for each feature (outside plotting loop)
for feature in data.columns:
  print(f"Value Counts for {feature}:")
  print(data[feature].value_counts())
  print("\n")  # Add a newline for readability

# Adjust layout for subplots
fig.tight_layout()  # Adjust spacing for subplots

# Show the plot
plt.show()


# In[7]:


nrows = 4  # Number of rows for 11 variables
ncols = 3  # Number of columns
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))

for idx, feature in enumerate(data.columns):
  ax = axes[idx // ncols, idx % ncols]

  # Dot plot with Seaborn
  sns.kdeplot(data=data, x=feature, shade=True, ax=ax)  # Use kdeplot for dot plots

  # Customize the plot (optional)
  ax.set_title(f"Distribution of {feature} (Dot Plot)")
  ax.set_xlabel(feature)
  ax.set_ylabel("Density")  # Adjust y-label for dot plots

  # Optional: Remove x-axis tick labels for subplots (except bottom row)
  if idx // ncols != nrows - 1:
    ax.tick_params(bottom=False)

fig.tight_layout()  # Adjust spacing for subplots
plt.show()


# In[8]:


# Calculate the correlation matrix
corr_matrix = data.corr()

# Print the correlation matrix (optional)
print("Correlation Matrix:")
print(corr_matrix.to_string())  # Or print(corr_matrix) for a full view

# Create the heatmap
plt.figure(figsize=(15, 8))
plt.title("Correlation of Dataset Features")
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the heatmap
plt.tight_layout()
plt.show()


# In[9]:


# Separate features and target variable
X = data.drop("HDA", axis=1) 
y = data["HDA"]


# In[10]:


# Perform Chi-square test
chi2_results = chi2(X, y)

# Set significance level
significance_level = 0.05

# Select features based on p-value
selected_features = chi2_results[1][chi2_results[1] <= significance_level].tolist()

# Create results DataFrame
results_df = pd.DataFrame({'feature': X.columns, 'chi2': chi2_results[0], 'p-value': chi2_results[1]})

# Sort results_df by chi2 values (descending order)
results_df.sort_values('chi2', ascending=False, inplace=True)

# Create the bar chart with customizations
chi_values = pd.Series(chi2_results[0], index=X.columns)
chi_values.sort_values(ascending=False, inplace=True)
chi_values.plot.bar(0)

# Customize plot elements (optional)
ax.set_title('Chi-Square Scores by Feature')  # Add a title
ax.set_xlabel('Feature')  # Set x-axis label
ax.set_ylabel('Chi-Square Score')  # Set y-axis label
plt.grid(axis='y')  # Add gridlines for better readability

# Print results (optional)
print(results_df.to_string(index=False))

plt.show()


# In[11]:


# Separate features (X) and target variable (y)
X = data.drop('HDA', axis=1)  # Replace 'target_variable' with your actual target variable name
y = data['HDA']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')  # Use liblinear solver for efficiency

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Access coefficients (feature weights)
coefficients = model.coef_.flatten()

# Print feature names and coefficients
feature_names = X.columns
for feature, coef in zip(feature_names, coefficients):
  print(f"{feature}: {coef:.4f}")

# Create a bar chart of feature coefficients
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title("Logistic Regression Feature Coefficients")
plt.tight_layout()
plt.show()


# ## Random Forest MODEL

# In[12]:


# Split data into training and testing sets
features = ['HBP', 'HBC', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'Sex', 'Age'] 
X = data[features]  # Select features
y = data['HDA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)


# In[14]:


y_pred = rf_model.predict(X_test)


# In[15]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", f"{accuracy*100:.2f}%")
print("Precision:", f"{precision*100:.2f}%")
print("Recall:", f"{recall*100:.2f}%")
print("F1-score:", f"{f1*100:.2f}%")

