# Building a classification model for Wine data set 

### Purpose:
by using a wine data set where the independent features are physicochemical attributes of the wine and the dependent variable is the wine's quality
We set out to build a classification model that would classify the wine by good or bad quality. 

#  Processing the data

1. Converting the quality variable from one that varies from 1 to 9, to a binary variable(0-1)
2. Removing highly correlated features
3. Oversampling the data using the SMOTE algorithm to rebalance the dataset
4. Scaling the data 

# Trying out various classification models 

various Supervised Learning models were tested on the dataset and Grid Search was used to find the best parameters for all of them. 
The model that preformed best was Random Forest with an 89% accuracy 

![graph of model accuracy](Readimages/1.jpg)

# Utilizing the model

After the best model was found and trained, a flask web app was built which predicts the quality of
new wines using the random forest model and returns the probabilty that a given wine is one of quality. 

![Entering the details of a wine in web app](Readimages/2.jpg)

![Results of prediction](Readimages/3.jpg)
