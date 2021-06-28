# Buliding a classification model for Wine dataset 

### Purpose:
by using a wine dataset where the independent features are physicochemical attributes of the wine and the dependent variable is the wine's quality
We set out to build a classification model that would classify the wine by good or bad quality. 

#  Processing the data

1. Coverting the quality variable from one that varies from 1 to 9, to a binary variable(0-1)
2. Removing highly correlated features
3. Scaling the data 

# Trying out various classification models 

various Supervised Learning models were tested on the dataset and Grid Search was used to find the best parameters for all of them. 
The model that preformed best was Random Forest with an 89% accuracy 

# Utilizing the model

After the best model was found and trained, a flask web app was built which analyzes new wine examples using the random forest model and returns
the probabilty that the wine is good.



