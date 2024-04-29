#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.exceptions import ConvergenceWarning  # Import for handling warnings
import warnings 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.optimizers import schedules
from tabulate import tabulate
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


data = pd.read_csv("cleaned_brfss.csv")


# Exploratory Data Analysis (EDA) and Feature Selection Using Chi2 and Logistic Regression

# In[3]:


data.describe()


# In[4]:


# Set display options for better presentation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Adjust width for better readability

# Print data information and missing value table
print("Data Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull())


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


# Separate features and target variable
X = data.drop("HDA", axis=1) 
y = data["HDA"]


# In[9]:


# Perform Chi-square test for feature selection
chi2_results = chi2(X, y)

# Set a significance level for feature selection
significance_level = 0.05

# Select features based on p-value
selected_features = chi2_results[1][chi2_results[1] <= significance_level].tolist()  # Access p-values using index 1

# Create DataFrame to store results
results_df = pd.DataFrame({'feature': X.columns, 'chi2': chi2_results[0], 'p-value': chi2_results[1]})

# Sort results_df by chi2 values (descending order)
results_df.sort_values('chi2', ascending=False, inplace=True)

# Plot chi-square values (from results_df) as a bar chart
results_df['chi2'].plot.bar(0)  # Access chi2 column from results_df

# Plot chi-square values
chi2_plot = results_df['chi2'].plot.bar(0)

#Print results in a formatted table
print(results_df.to_string(index=False))

# Set custom labels for x-axis (feature names)
plt.xticks(ticks=range(len(X.columns)), labels=X.columns)
plt.show()  # Add plt.show() to display the plot


# In[10]:


target_variables = ['HBP', 'HBC', 'BMI', 'Smoker', 'Stroke', 'Diabetes']


# In[11]:


def check_imbalance(data, target_variables):
  """
  Checks for class imbalance in a DataFrame for multiple target variables.

  Args:
      data (pandas.DataFrame): The DataFrame containing the data.
      target_variables (list): A list of target variable names.

  Prints information about class imbalance for each target variable,
  handling potential missing columns and empty target_variables lists.
  """

  # Check if data is a DataFrame
  if not isinstance(data, pd.DataFrame):
      print("Error: 'data' must be a pandas DataFrame.")
      return

  # Handle potential missing columns and empty target_variables list
  if not target_variables:
      print("No target variables provided.")
      return

  existing_targets = [target for target in target_variables if target in data.columns]
  if not existing_targets:
      print("None of the target variables exist in the data.")
      return

  for target_variable in existing_targets:
      class_counts = data[target_variable].value_counts()
      total_count = len(data)

      # Print class imbalance information
      print(f"Target: {target_variable}")
      for target, count in class_counts.items():
          ratio = count / total_count
          print(f"  Count: {count}, Ratio: {ratio:.2f}")
      print()  # Add a blank line between target variables

# Example usage (assuming you have data and target_variables defined)
check_imbalance(data, target_variables)


# In[12]:


# Stratified ShuffleSplit for maintaining class distribution during splitting
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# In[13]:


train_indices, test_indices = next(iter(sss.split(X, y)))
train_index = pd.Index(train_indices)  # Convert to pandas Index


# In[14]:


def undersample_data(X, y, target_variables):
    undersampled_features = pd.DataFrame()
    undersampled_targets = pd.DataFrame()

    for var in target_variables:
        # Check if the target variable is present in the dataframe columns
        if var in X.columns:
            # Handle missing values in y (optional)
            # You can use techniques like forward fill or imputation
            y_filtered = y[y[var].notna()]  # Filter y for non-missing values in 'var' column
            # Perform undersampling for the target variable
            undersampled_features = pd.concat([undersampled_features, X[var]], axis=1)
            undersampled_targets[var] = y_filtered[var]
        else:
            print(f"Warning: {var} not found in dataframe columns.")

    return undersampled_features, undersampled_targets


# In[15]:


# Select features and scale them
X_reduced = X[target_variables]  # Select only target features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Set hyperparameters
max_iter = 200  # Maximum iterations
C = 0.1  # Regularization parameter (not used in manual training)
learning_rate = 0.1
decay_rate = 0.97  # Learning rate decay rate (not used in manual training)
decay_steps = 20  # Steps for learning rate decay (not used in manual training)

# Create manual learning rate schedule
learning_rates = np.linspace(learning_rate, learning_rate * decay_rate ** (max_iter // decay_steps), max_iter)

# Initialize weights and intercept
weights = np.zeros(X_scaled.shape[1])  # Weights size based on selected features
intercept = 0.0

# Define table data (list of lists)
table_data = []

# Print header
print("Logistic Regression Training Progress:")

# **Define the LogisticRegression model**
# We don't set hyperparameters here as manual updates are used
model = LogisticRegression(solver='liblinear')  # Use liblinear solver for efficiency

# Train model with manual updates
for epoch in range(max_iter):
  current_learning_rate = learning_rates[epoch]

  # Fit the model using only relevant features
  model.fit(X_scaled, y)

  # Access loss using decision_function (after fitting)
  decision_scores = model.decision_function(X_scaled)
  training_loss = np.mean(np.log(1 + np.exp(-decision_scores * y)))

  # Update weights and intercept manually
  weights = weights - current_learning_rate * np.dot(X_scaled.T, decision_scores)
  intercept = intercept - current_learning_rate * np.mean(decision_scores)

  # Append data to table with 4 decimal places
  table_data.append([epoch + 1, f"{current_learning_rate:.4f}", f"{training_loss:.4f}"])

# Print table using tabulate
print(tabulate(table_data, headers=["Epoch", "Learning Rate", "Training Loss"]))

# Access coefficients and intercept after manual training
coefficients = weights
intercept = intercept


# In[16]:


print("Feature Coefficients:")
coefficients = model.coef_.flatten()  # Extract coefficients

# Select target features
target_features = ['HBP', 'HBC', 'BMI', 'Smoker', 'Stroke', 'Diabetes']

# Filter features and coefficients based on target_variables
sorted_features_coefs = []
for feature, coef in zip(X.columns, coefficients):
  if feature in target_features:
    sorted_features_coefs.append((feature, coef))
    print(f"{feature}: {coef:.4f}") 

# Sort features and coefficients together by coefficient value (absolute value)
sorted_features_coefs = sorted(sorted_features_coefs, key=lambda x: abs(x[1]), reverse=True)

# Extract data for visualization
feature_names = [f[0] for f in sorted_features_coefs]  # Feature names
coef_values = [abs(f[1]) for f in sorted_features_coefs]  # Absolute coefficient values
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_names, y=coef_values)
plt.xlabel("Features")
plt.ylabel("Absolute Coefficient Value")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title("Feature Coefficients (Target Variables)")
plt.tight_layout()
plt.show()


# K-Nearest Neighbor Model

# In[18]:


# Split data into training and testing sets
features = ['HBP', 'HBC', 'BMI', 'Smoker', 'Stroke', 'Diabetes'] 
X = data[features]  # Select features
y = data['HDA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


# In[20]:


y_pred = knn_model.predict(X_test)


# In[23]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", f"{accuracy*100:.2f}%")
print("Precision:", f"{precision*100:.2f}%")
print("Recall:", f"{recall*100:.2f}%")
print("F1-score:", f"{f1*100:.2f}%")


# In[ ]:




