import seaborn as sns
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
df = sns.load_dataset("iris")
df.head()
df.info()
df.species.value_counts()
sns.pairplot(df, hue = "species")
# Convert categorical data to numerical data
species_mapping = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
df['species_num'] = df['species'].map(species_mapping)

# Display the updated DataFrame
df.head()
plt.figure(figsize = (15, 9))
df_num = df.drop(columns=['species'])
sns.heatmap(df_num.corr(), annot = True, cmap='coolwarm')
df.drop(["petal_length", "species_num"], axis = 1, inplace = True)
df.head()
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Separate features (X) and target variable (y)
X = df.drop(columns=["species"])
y = df["species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)
accuracy_score_dt = accuracy_score(y_test, y_pred)
print("Accuracy score : ", accuracy_score_dt) 
print("Criterion used: ", dt_model.criterion)
print("Maximum depth: ", dt_model.max_depth)
print("Maximum number of features considered: ", dt_model.max_features)
print("Minimum samples split: ", dt_model.min_samples_split)
print("Minimum samples leaf: ", dt_model.min_samples_leaf)
# Cross-validation for model evaluation
from sklearn.model_selection import cross_val_score
cv_accuracy_scores = cross_val_score(estimator=dt_model, X=X_train, y=y_train, cv=10)
cv_accuracy_scores_dt = cv_accuracy_scores.mean()
print("Mean of Accuracy score : ", cv_accuracy_scores_dt) 
print(confusion_matrix(y_test, y_pred))
# Visualize confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
targets = ["setosa", "versicolor", "virginica"]
sns.heatmap(cnf_matrix, annot=True, cmap='YlGnBu',xticklabels=targets, yticklabels=targets)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()
print(classification_report(y_test, y_pred))
from sklearn import tree
features = list(df.columns[:-1]) # select all elements of a list except for the last one (target).
print(features)
print(targets)
# Plot the decision tree model
plt.figure(figsize = (20, 10))
tree_viz = tree.plot_tree(dt_model, filled=True, feature_names=features, class_names=targets)
from sklearn.model_selection import GridSearchCV
dt_model2 = DecisionTreeClassifier(random_state=0)

# Defining a dictionary containing the hyperparameters to be tuned
dt_params = {"criterion" : ["gini", "entropy"], 
             'max_depth': [None, 5, 10, 15],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
            'max_features': [None,"auto", "sqrt", "log2"]}

# Creating a GridSearchCV model with 10-fold cross-validation
dt_cv_model = GridSearchCV(estimator=dt_model2, param_grid=dt_params, cv=10)

# Fitting the GridSearchCV model to the training data
dt_cv_model.fit(X_train, y_train)

# Extracting the best hyperparameters found by GridSearchCV
dt_cv_model.best_params_
dt_tuned = DecisionTreeClassifier(criterion="gini", max_depth=None, max_features=None,
                                  min_samples_leaf=2, min_samples_split=2)
dt_tuned.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tuned = dt_tuned.predict(X_test)
accuracy_score_dt_tuned = accuracy_score(y_test, y_pred_tuned)
print("Accuracy score : ", accuracy_score_dt_tuned) 
cv_accuracy_scores = cross_val_score(estimator=dt_tuned, X=X_train, y=y_train, cv=10)
cv_accuracy_scores_dt_tuned = cv_accuracy_scores.mean()
print("Mean of Accuracy score : ", cv_accuracy_scores_dt_tuned) 
performance = pd.DataFrame({
    'Metrics': ['accuracy', 'CV accuracy'],
    'DT': [accuracy_score_dt, cv_accuracy_scores_dt],
    'DT tuned': [accuracy_score_dt_tuned, cv_accuracy_scores_dt_tuned]
})
print(performance.set_index('Metrics'))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score_rf = accuracy_score(y_test, y_pred)
print("Accuracy score : ", accuracy_score_rf) 
print("n_estimators used: ", rf_model.n_estimators)
print("Maximum depth: ", dt_model.max_depth)
print("Maximum number of features considered: ", dt_model.max_features)
print("Minimum samples split: ", dt_model.min_samples_split)
print("Minimum samples leaf: ", dt_model.min_samples_leaf)
cv_accuracy_scores = cross_val_score(estimator=rf_model, X=X_train, y=y_train, cv=10)
cv_accuracy_scores_rf = cv_accuracy_scores.mean()
print("Mean of Accuracy score : ", cv_accuracy_scores_rf) 
rf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(rf_matrix, annot=True, cmap='YlGnBu',xticklabels=targets, yticklabels=targets)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
print(classification_report(y_test, y_pred))
rf_model.estimators_
plt.figure(figsize = (20, 10))
tree_rf = tree.plot_tree(rf_model.estimators_[0], filled=True, feature_names=features, class_names=targets)
rf_model2 = RandomForestClassifier(random_state=0)

# Defining a dictionary containing the hyperparameters to be tuned
rf_params = {"n_estimators" : [None, 50, 100, 300],
             'max_depth': [None, 3, 5, 7],
             'min_samples_split': [2, 4, 6],
            'max_features': [None, "sqrt", "log2"]}

# Creating a GridSearchCV model with 10-fold cross-validation
rf_cv_model = GridSearchCV(estimator=rf_model2, param_grid=rf_params, cv=10, n_jobs=-1)
# n_jobs=-1 utilize all available CPU cores to perform the computations in parallel

# Fitting the GridSearchCV model to the training data
rf_cv_model.fit(X_train, y_train)

# Extracting the best hyperparameters found by GridSearchCV
rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(max_depth=3, max_features=None, min_samples_split=2, n_estimators=50)
rf_tuned.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_score_rf_tuned = accuracy_score(y_test, y_pred_tuned)
print("Accuracy score : ", accuracy_score_rf_tuned) 
cv_accuracy_scores = cross_val_score(estimator=rf_tuned, X=X_train, y=y_train, cv=10)
cv_accuracy_scores_rf_tuned = cv_accuracy_scores.mean()
print("Mean of Accuracy score : ", cv_accuracy_scores_rf_tuned) 
rf_tuned_matrix = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(rf_tuned_matrix, annot=True, cmap='YlGnBu',xticklabels=targets, yticklabels=targets)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
print(classification_report(y_test, y_pred_tuned))
performance = pd.DataFrame({
    'Metrics': ['accuracy', 'CV accuracy'],
    'RF': [accuracy_score_rf, cv_accuracy_scores_rf],
    'RF tuned': [accuracy_score_rf_tuned, cv_accuracy_scores_rf_tuned]
})
print(performance.set_index('Metrics'))
feature_importances = pd.DataFrame(rf_tuned.feature_importances_, index=X.columns, columns=["Weight"]).sort_values(by="Weight", ascending=False)
feature_importances
sns.barplot(x="Weight", y=feature_importances.index, data=feature_importances, palette="viridis")
plt.xlabel("Feature Importance Weight")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()
plt.figure(figsize = (10, 10))
tree_rf = tree.plot_tree(rf_tuned.estimators_[0], filled=True, feature_names=features, class_names=targets)

dt_tuned.fit(X_train, y_train)
rf_tuned.fit(X_train, y_train)
performance = pd.DataFrame({
    'Metrics': ['accuracy', 'CV accuracy'],
    'DT tuned': [accuracy_score_dt_tuned, cv_accuracy_scores_dt_tuned],
    'RF tuned': [accuracy_score_rf_tuned, cv_accuracy_scores_rf_tuned]
})
print(performance.set_index('Metrics'))
dt_time = timeit.timeit(lambda: dt_tuned.fit(X_train, y_train), number=100)
rf_time = timeit.timeit(lambda: rf_tuned.fit(X_train, y_train), number=100)

print(f"Mean time per run (Decision Tree): {dt_time / 100 * 1000:.3f} ms")
print(f"Mean time per run (Random Forest): {rf_time / 100 * 1000:.3f} ms") 
