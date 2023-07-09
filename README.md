Dataset source: [Kaggle](https://www.kaggle.com/)

# Problem Statement:

Credit risk as the board in banks basically centers around deciding the probability of a customer’s default or credit decay and how expensive it will end up being assuming it happens. It is important to consider major factors and predict beforehand the probability of consumers defaulting given their conditions.
Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low-income levels, or too many inquiries on an individual’s credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!).

# Objective:

We want to automate this task of prediction with the power of machine learning (I have used Logistic Regression) and pretty much every commercial bank does so nowadays. In this project, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.

# Procedure

**Step 1** : Importing the pandas package and loading the dataset
On observing, the output appears a bit confusing at its first sight, but let’s try
to figure out the most important features of a credit card application. We find
that since the data is confidential, the contributor of this dataset has
anonymized the feature names. The features of this dataset have been
anonymized to protect the privacy, but this blog gives us a pretty good
overview of the probable features.
Columns are:
Gender, Age, Debt, Married, BankCustomer,
EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed,
CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the
ApprovalStatus.

**Step 2** : As we can see from our first output at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some pre-processing, therefore we start manipulating the data.

We’ve uncovered some issues that will affect the performance of our machine learning model if they go unchanged:

● Our dataset contains both numeric and non-numeric data (specifically data that are of float64, int64 and object types). Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64,float64, int64 and int64 respectively) and all the other features contain non-numeric values (of type object).

● The dataset also contains values from several ranges. Some features have a value range of 0–28, some have a range of 2–67, and some have a range of 1017–100000. Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.

● Finally, the dataset has **missing values**, which we’ll take care of in this task. The missing values in the dataset are labeled with ‘?’, which can be seen in the last cell’s output.

We are going to impute the missing values with a strategy called mean imputation.

**Step 3** : Pre-processing the data

There is still some minor but essential data pre-processing needed before we proceed towards building our machine learning model. We are going to divide these remaining pre-processing steps into three main tasks:
1. Convert the non-numeric data into numeric.
2. Split the data into train and test sets.
3. Scale the feature values to a uniform range.

![](https://github.com/harshvardhan2511/Credit-Card-Approval-Prediction/blob/main/pairplot.png)
![](https://github.com/harshvardhan2511/Credit-Card-Approval-Prediction/blob/main/Heatmap.png)
![](https://github.com/harshvardhan2511/Credit-Card-Approval-Prediction/blob/main/Boxplot.png)

**Step 4** : Splitting the dataset into training and test sets

**Step 5** : Fitting a Logistic Regression Model to the training set

**Step 6** : Making predictions and evaluating the performance of the model and Grid Search and making the model perform better. Our model was pretty good! It was able to yield an accuracy score of almost **84%**



