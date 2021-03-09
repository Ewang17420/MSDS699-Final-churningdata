# MSDS699-Final-churningdata
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ewang17420/MSDS699-Final-churningdata)

This is a project for building a ML model to predict churning customer from a bank dataset.

## Data set description

source: https://www.kaggle.com/sakshigoyal7/credit-card-customers <br />
Here is the description from the Kaggle owner:  
"A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.  

I got this dataset from a website with the URL as https://leaps.analyttica.com/home. I have been using this for a while to get datasets and accordingly work on them to produce fruitful results. The site explains how to solve a particular business problem.  

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.  

We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers."  

## Feature description

There are 23 columns in our dataset, 1 is for label, and other 22 is for features.  <br />

- CLIENTNUM -integer. <br />
Client number. Unique identifier for the customer holding the account. <br />

- Attrition_Flag -string. <br />
Internal event (customer activity) variable - if the account is closed then 1 else 0. <br />

- Customer_Age -integer. <br />
Demographic variable - Customer's Age in Years. <br />

- Gender -string. <br />
Demographic variable - M=Male, F=Female. <br />

- Dependent_count -integer   <br />
Demographic variable - Number of dependents. <br />

- Education_Level -string. <br />
Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.). <br />

- Marital_Status -string. <br />
Demographic variable - Married, Single, Divorced, Unknown. <br />

- Income_Category -string. <br />
Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, >. <br />

- Card_Category -string. <br />
Product Variable - Type of Card (Blue, Silver, Gold, Platinum). <br />

- Months_on_book -integer  <br />
Period of relationship with bank. <br />

- Total_Relationship_Count -integer. <br />
Total no. of products held by the customer. <br />

- Months_Inactive_12_mon -integer. <br />
No. of months inactive in the last 12 months. <br />

- Contacts_Count_12_mon -integer.  <br />
No. of Contacts in the last 12 months. <br />

- Credit_Limit -integer. <br />
Credit Limit on the Credit Card. <br />

- Total_Revolving_Bal -integer. <br />
Total Revolving Balance on the Credit Card. <br />

- Avg_Open_To_Buy -integer. <br />
Open to Buy Credit Line (Average of last 12 months). <br />

- Total_Amt_Chng_Q4_Q1 -float. <br />
Change in Transaction Amount (Q4 over Q1). <br />

- Total_Trans_Amt -integer. <br />
Total Transaction Amount (Last 12 months). <br />

- Total_Trans_Ct -integer. <br />
Total Transaction Count (Last 12 months). <br />

- Total_Ct_Chng_Q4_Q1 -float. <br />
Change in Transaction Count (Q4 over Q1). <br />

- Avg_Utilization_Ratio -float. <br />
Average Card Utilization Ratio. <br />

#the following columns can be omitted. <br />

- Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1. <br />
Naive Bayes. <br />

- Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2. <br />
Naive Bayes. <br />

## Method description

First of all, let's deal with preprocessing.
By the magic of imputer, standard scaler and one hot encoder, we can get a clean dataset and ready to be analyzed.
Then from the data visualiztion, we can find out that this is an imbalanced dataset. 
So after preprocessing I decided to use PCA to reduce dimension of the dataset and SMOTE to resampling.
Last, random search can help us to find the best model and it's hyperparameters to fit the data.

## Result summary

After running the pipeline, the Gradient Boosting Classfier is the best model and the accuracy can reach 0.95.
