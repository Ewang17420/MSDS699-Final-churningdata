# MSDS699-Final-churningdata
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ewang17420/MSDS699-Final-churningdata)

This is a project for building a ML model to predict churning customer from a bank dataset.

### data set description

source: https://www.kaggle.com/sakshigoyal7/credit-card-customers. 
Here is the description from the Kaggle owner:
"A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.  

I got this dataset from a website with the URL as https://leaps.analyttica.com/home. I have been using this for a while to get datasets and accordingly work on them to produce fruitful results. The site explains how to solve a particular business problem.  

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.  

We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers."  

### feature description

There are 23 columns in our dataset, 1 is for label, and other 22 is for features.  

- CLIENTNUM -integer. 
Client number. Unique identifier for the customer holding the account. 

- Attrition_Flag -string. 
Internal event (customer activity) variable - if the account is closed then 1 else 0. 

- Customer_Age -integer. 
Demographic variable - Customer's Age in Years. 

- Gender -string. 
Demographic variable - M=Male, F=Female. 

- Dependent_count -integer   
Demographic variable - Number of dependents. 

- Education_Level -string. 
Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.). 

- Marital_Status -string. 
Demographic variable - Married, Single, Divorced, Unknown. 

- Income_Category -string. 
Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, >. 

- Card_Category -string. 
Product Variable - Type of Card (Blue, Silver, Gold, Platinum). 

- Months_on_book -integer  
Period of relationship with bank. 

- Total_Relationship_Count -integer. 
Total no. of products held by the customer. 

- Months_Inactive_12_mon -integer. 
No. of months inactive in the last 12 months. 

- Contacts_Count_12_mon -integer.  
No. of Contacts in the last 12 months. 

- Credit_Limit -integer. 
Credit Limit on the Credit Card. 

- Total_Revolving_Bal -integer. 
Total Revolving Balance on the Credit Card. 

- Avg_Open_To_Buy -integer. 
Open to Buy Credit Line (Average of last 12 months). 

- Total_Amt_Chng_Q4_Q1 -float. 
Change in Transaction Amount (Q4 over Q1). 

- Total_Trans_Amt -integer. 
Total Transaction Amount (Last 12 months). 

- Total_Trans_Ct -integer. 
Total Transaction Count (Last 12 months). 

- Total_Ct_Chng_Q4_Q1 -float. 
Change in Transaction Count (Q4 over Q1). 

- Avg_Utilization_Ratio -float. 
Average Card Utilization Ratio. 

#the followinf columns can be omitted. 

- Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1. 
Naive Bayes. 

- Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2. 
Naive Bayes. 

### Method description

First of all, let's deal with preprocessing.
By the magic of imputer, standard scaler and one hot encoder, we can get a clean dataset and ready to be analyzed.
Then from the data visualiztion, we can find out that this is an imbalanced dataset. 
So after preprocessing I decided to use PCA to reduce dimension of the dataset and SMOTE to resampling.
Last, random search can help us to find the best model and it's hyperparameters to fit the data.

### result summary

After running the pipeline, the Gradient Boosting Classfier is the best model and the accuracy can reach 0.95.
