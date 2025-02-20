#!/usr/bin/env python
# coding: utf-8

# #  <font color=green>Credit Card Fraud Detection Project</font>

# Hello!
# 
# I am really excited about machine-learning and decided to take on this project as the first of many to get more comfortable the models used. 
# 
# This project covers credit card fraud and is meant to look at a dataset of transactions and predict whether it is fraudulent or not. I learned alot of this from 
# Eduonix Learning Solutions. 

# ## Imports 

# In[1]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn


# ## Data Importing

# In[2]:


path = 'creditcard.csv'


# In[3]:


data = pd.read_csv(path)


# ## Exploring the Dataset

# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


print(data.columns)


# In[7]:


data.shape


# In[8]:


# randomly selected 20% of the dataframe rows (fix the seed at 1)
df = data.copy() # storing this for later use
data = data.sample(frac = 0.2, random_state = 1)
print(data.shape)


# In[9]:


# plot the histogram of each parameter
data.hist(figsize = (20, 20),grid=False)
plt.show()


# You can see most of the V's are clustered around 0 with some or no outliers. Notice we have very few fraudulent cases over valid cases in our class histogram.

# In[10]:


# determine the number of fraud cases


def compute_outlier_fraction(data):
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlier_fraction = len(fraud) / float(len(valid))
    return outlier_fraction
    

# percentage_fraud = (len(fraud) / len(data) ) *100
outlier_fraction = compute_outlier_fraction(data)
# print(f'Percentage fraud : {percentage_fraud} %\n')
print(f'Outlier_fraction : {outlier_fraction}')

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
print(f'Fraud Cases: {len(fraud)}')
print(f'Valid Cases: {len(valid)}')


# ## Visualization of features

# In[11]:


import matplotlib.pyplot as plt

def compare_histograms(a, b):
    """Plots histograms of each feature in DataFrames a and b on the same axes."""
    
    # List of columns with 'numerical values' only
    features = a.select_dtypes(include=['number']).columns

    # Create subplots
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(5, 3 * len(features)))
 
 

    # Loop through each feature and plot histograms
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(a[feature], bins=30, color='red', alpha=0.5, label='Fraud', density=True)
        ax.hist(b[feature], bins=30, color='blue', alpha=0.5, label='Valid', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.show()

# Call the function to compare histograms
compare_histograms(fraud, valid)


# In[12]:


from sklearn.ensemble import RandomForestClassifier

# Define X and Y


# Training Dataframe
X = data.drop(columns='Class')

# Target ( i.e 'Class')
Y = data['Class']

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, Y)

# Feature importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display results
# print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'],color='magenta')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()


# In[13]:


'''Recall that in supervised learning, the label is a target not a feature
Thus I will drop the 'Class' column when plotting the correlation matrix '''
# correlation matrix
corrmat = data.drop(columns=['Class']).corr()  #
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# You can see a lot of the values are close to 0 . Most of them are fairly unrelated. The lighter squares signify a stronger correlation. 

# ## Organizing the Data

# In[14]:


'''Since, as mentioned 'Class' is a target, we drop it when training (For X) '''

# Training Dataframe
X = data.drop(columns='Class')

# Target ( i.e 'Class')
Y = data['Class']

print(X.shape)
print(Y.shape)


# In[15]:


X.columns # To confirm 'Class' is absent from our features


# #  <font color=red>Applying Algorithms (Machine Learning)</font>

# In[21]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# # First, a function that will allow an ouput dataframe for plotting

# In[22]:


import pandas as pd
from sklearn.metrics import confusion_matrix


def create_verdict_dataframe(X, Y, y_pred):
    """
    Creates a DataFrame containing training features and a 'verdict' column 
    that classifies each transaction as TN, TP, FN, or FP.

    Parameters:
    - X: Feature DataFrame used for training
    - Y: Actual labels (0 = valid, 1 = fraud)
    - y_pred: Model predictions (0 = valid, 1 = fraud)

    Returns:
    - DataFrame with all features and a new 'verdict' column
    """
    
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()

    # Define verdict labels for each row
    def get_verdict(actual, predicted):
        if actual == 1 and predicted == 1:
            return "TP"  # True Positive (Correctly detected fraud)
        elif actual == 1 and predicted == 0:
            return "FN"  # False Negative (Missed fraud)
        elif actual == 0 and predicted == 1:
            return "FP"  # False Positive (Normal wrongly flagged as fraud)
        else:
            return "TN"  # True Negative (Correctly classified as normal)

    # Create DataFrame with predictions and actual labels
    df_result = X.copy()
    df_result['Actual'] = Y
    df_result['Predicted'] = y_pred
    df_result['Verdict'] = [get_verdict(a, p) for a, p in zip(Y, y_pred)]

    return df_result




# # A function that will visualize the outliers on a scattered plot for dominant features

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_fraud_detection(df_verdict, feature1='V14', feature2='V17'):
    """
    Creates side-by-side scatter plots for fraud detection based on two selected features.
    
    Parameters:
    - df_verdict: DataFrame containing transaction data with a 'Verdict' column
    - feature1: First feature to visualize on the y-axis
    - feature2: Second feature to visualize on the y-axis
    """
    
    # Mapping verdicts to colors
    color_map = {"TP": "red", "FP": "yellow", "TN": "green", "FN": "black"}
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    # Scatter plot for Feature 1
    sns.scatterplot(x=df_verdict.index, y=df_verdict[feature1], hue=df_verdict['Verdict'], 
                    palette=color_map, s=50, ax=axes[0])
    axes[0].set_title(f'Visualization of {feature1}')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel(feature1)
    axes[0].legend(title='Verdict', loc='best')

    # Scatter plot for Feature 2
    sns.scatterplot(x=df_verdict.index, y=df_verdict[feature2], hue=df_verdict['Verdict'], 
                    palette=color_map, s=50, ax=axes[1])
    axes[1].set_title(f'Visualization of {feature2}')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel(feature2)
    axes[1].legend(title='Verdict', loc='best')

    # Show the plot
    plt.show()




# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from termcolor import colored




# The ML function

def run_ml(X, Y, outlier_fraction, state=1):
    
    # A pre-legend/explanation
    print(colored("True Positive (TP)", "red") + "  : Case correctly identified as fraud")
    print(colored("False Positive (FP)", "yellow") + " : Normal transaction incorrectly flagged as fraud")
    print(colored("True Negative (TN)", "green") + "  : Normal transaction correctly classified as valid")
    print(colored("False Negative (FN)", "grey") + " : Fraud case missed and classified as normal\n")
    print('-' * 80)
    print('-' * 80)
    
    classifiers = {
        'Isolation Forest': IsolationForest(max_samples=len(X),  
                                           contamination=outlier_fraction,  
                                           random_state=state),
        'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction),
        'Support Vector Machine (SVM)': OneClassSVM(kernel='rbf', nu=0.005, gamma='scale')
    }

    for clf_name, clf in list(classifiers.items())[::-1]:
        
        if clf_name == 'Support Vector Machine (SVM)':
            scaler = StandardScaler()  # SVM requires feature scaling
            X_scaled = scaler.fit_transform(X)
            clf.fit(X_scaled)
            y_pred = clf.predict(X_scaled)

        elif clf_name == 'Local Outlier Factor':
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_

        else:  # Isolation Forest
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)

        # 🔹 Convert predictions: 1 → 0 (valid), -1 → 1 (fraud)
        y_pred = np.where(y_pred == 1, 0, 1)

        # 🔹 Create the DataFrame with verdict labels
        df_verdict = create_verdict_dataframe(X, Y, y_pred)

        # 🔹 Plot results
        plot_fraud_detection(df_verdict)

        # 🔹 Compute classification report once
        report = classification_report(Y, y_pred, output_dict=True)

        # Extract key metrics
        fraud_recall = report['1']['recall'] * 100   # % of fraud correctly identified
        fraud_precision = report['1']['precision'] * 100  # % of flagged fraud actually fraud
        fraud_f1 = report['1']['f1-score'] * 100  # F1 score for fraud detection
        false_negatives = 100 - fraud_recall  # Fraud cases missed

        # 🔹 Correct False Positive Rate calculation
        tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
        false_positive_rate = (fp / (fp + tn)) * 100  # Normal cases wrongly flagged as fraud

        # Print performance metrics
        print(f'The model {clf_name} has {np.sum(y_pred != Y)} errors')
        print(f'Accuracy: {accuracy_score(Y, y_pred) * 100:.4f} %\n')
        print(classification_report(Y, y_pred))
        print(f"{fraud_recall:.3f}% of fraud cases were accurately identified as fraud.")
        print(f"{false_negatives:.3f}% of fraud cases were missed.")
        print(f"{false_positive_rate:.3f}% of normal transactions were incorrectly flagged as fraud.")
        print('-' * 80)


# ## Fit the Model

# In[27]:


run_ml(X,Y,outlier_fraction)


# Looking at precision for fraudulent cases (1) lets us know the percentage of cases that are getting correctly labeled. 'Precision' accounts for false-positives. 'Recall' accounts for false-negatives. Low numbers could mean that we are constantly calling clients asking them if they actually made the transaction which could be annoying.
# 
# Goal: To get better percentages.

# Our Isolation Forest method (which is Random Forest based) was able to produce a better result. Looking at the f1-score 26% (or approx. 30%) of the time we are going to detect the fraudulent transactions.

# #  <font color=blue>Optimizing models : Using less features of higher importance</font>

# In[28]:


# first reducing the features

reduced_features = ['V12','V14', 'V11', 'V17', 'V10', 'V16' ,'V9', 'V18']

new_sample = df.sample(frac = 0.2, random_state = 2)

of_reduced = compute_outlier_fraction(new_sample)

X_reduced = new_sample [reduced_features]

Y_reduced = new_sample['Class']

print(X_reduced.shape)
print(Y_reduced.shape)


# In[29]:


run_ml(X_reduced,Y_reduced,of_reduced)


# # In this project, we focused on using "class" as our target feature because it simplifies the classification of fraudulent and valid transactions. We employed feature importance analysis to identify the most relevant features for our models, selecting the most impactful ones for optimization. Our results indicate that the Support Vector Machine (SVM) consistently outperformed the other models, both before and after optimization, while the Local Outlier Factor (LOF) exhibited the weakest performance. Interestingly, we observed that the LOF model is particularly effective at identifying normal transactions. This raises an important question: Is this behavior driven by the hyperparameters we used or by the selected features? Further investigation is required to determine the underlying cause. 

# In[ ]:




