# Task 34: Regression Predictions

## Objective
Understand how a trained machine learning model generates predictions for new, unseen input values using Python, Pandas, and Scikit-learn.

## Project Overview
In this task, a Simple Linear Regression model is trained on a Salary vs. Experience dataset. Once trained, the model is used to perform inference on new years-of-experience values, displaying the predicted salaries in a structured table.

## Tools and Libraries Used
* **Python**: Core programming language
* **Pandas**: Data manipulation and structured tabular formatting
* **NumPy**: Numerical array operations
* **Scikit-learn**: Linear Regression model training and prediction

## Workflow and Implementation
1. **Dataset Creation**: Constructed a dataset containing 10 records mapping `Experience_Years` (feature) to `Salary` (target).
2. **Data Preparation**: Separated the dataset into a 2D feature matrix (`X`) and a 1D target vector (`y`).
3. **Model Training**: Initialized `LinearRegression()` from `sklearn.linear_model` and fitted it to the training data.
4. **Inference**: Passed a 2D array of new experience values (`2.5`, `4.5`, `6.5`, and `10.0` years) into `model.predict()` to generate salary estimates.
5. **Output Formatting**: Organized the new inputs and corresponding predictions into a clean Pandas DataFrame.

## Model Results

### Learned Parameters
* **Intercept (c)**: 27033.73
* **Coefficient (m)**: 9837.68

### Prediction Results for New Inputs

| Years of Experience | Predicted Salary (USD) |
| :--- | :--- |
| 2.5 | 51627.93 |
| 4.5 | 71303.29 |
| 6.5 | 90978.66 |
| 10.0 | 125410.54 |

*(Note: Passing a NumPy array to a model trained on a named Pandas DataFrame triggers a standard Scikit-learn `UserWarning` regarding feature names, which does not affect calculation accuracy. Using a DataFrame with matching column names for prediction resolves the warning.)*

## Interview Questions and Answers

**Q1: What does `model.predict()` do?**
* **Answer:** `model.predict()` takes input feature values and applies the parameters learned during training (weights/coefficients and bias/intercept) to calculate and return target output predictions. For simple linear regression, it evaluates the linear equation `y = mx + c`.

**Q2: What is inference in machine learning?**
* **Answer:** Inference is the process of using a trained machine learning model to generate predictions on new, unseen data points. Unlike training—where the model learns patterns from historical data—inference applies those established patterns in real-world operational scenarios.

**Q3: Why must prediction input have the correct shape?**
* **Answer:** Scikit-learn expects feature inputs to be 2-dimensional arrays structured as `(n_samples, n_features)`. Passing a 1D array or an incorrect number of feature columns causes a shape mismatch error because the underlying matrix operations require dimensions that match the feature space the model was trained on.
