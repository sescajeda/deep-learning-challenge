# deep-learning-challenge
# Goal: 
With knowledge of machine learning and neural networks, use the features in the provided dataset to create a binary classifier that can predict wheter applicants will be successful in funded by Alphabet Soup. 

# Repo Description: 
In this repo you will find Started_Code.ipynb file that contains the first model that was created for this data set.  In this first model the neural net that was created had 2 layers each with the number of nodes equal to the number of features the model was fed. This resulted in an accuracy for this model was below 0.75.
The Starter_Code_Optimization_1.ipynb contains the code for the first attempt to optimize the model by adding a 3rd layer, each of the 3 layers had the same number of nodels (number of features fed to the model). This model also had an accuracy below 0.75. 
The Starter_Code_Optimization_2.ipynb contains the code for the second attempt to optimize the model by keeping the 3 layers from the previous model as well as some feature engineering.  For this model the "INCOME_AMT" column which was a category was converted into a numerical feature by converting the intervals to a value from 1-8.  Even with those changes this model's accuracy was below 0.75. 
The Starter_Code_Optimization_3.ipynb contains the code for the third attempt to optimized the model by keeping everything for the second attempt as well as keeping the company names.  There were companies that applied multiple times under different circumstances so there appears to be value in feeding that model those names.  Companies that had a count below 50 were grouped into an "other" category.  These changes resulted in the model having an accuracy above 0.75.  

# Overview of the analysis: 
The goal of this analysis is to create a model that identifies applicants for funding with the best chance of success.  In order to create the model we will create a binary classifier that has above 0.75 accuracy.  

# Results: 
Data Preprocessing

What variable(s) are the target(s) for your model?
- IS_SUCCESSFUL—Was the money used effectively

What variable(s) are the features for your model?
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested

What variable(s) should be removed from the input data because they are neither targets nor features?
- In the initial models of this analysis both the EIN and NAME variable were removed because they appeared to be uncessery.  However, in the later models it was observed that the NAME variable could prove beneficial in training the model because there were institutions that had applied multiple times under different circumstances.  In the third model the NAME variable was kept as a feature but the EIN was still removed. 

Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
- There were 4 models in this analysis and the rule activation function was used throughout.  For each model the number of nodes (neurons) for each layer was set equal to the numer of input features that the model was fed.
- Model 1 (Starter_Code) had 2 hidden layers, each with 43 neurons (the number of features the model was fed).  
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 43)                  │           1,892 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 43)                  │           1,892 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │              44 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,828 (14.95 KB)
 Trainable params: 3,828 (14.95 KB)
 Non-trainable params: 0 (0.00 B)
- Model 2 (Starter_Code_Optimization_1) had 3 hidden layers, each with 43 neurons (the number of features the model was fed).  For this model an extra layer was added in a attempt to optimize the model. 
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 43)                  │           1,892 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 43)                  │           1,892 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 43)                  │           1,892 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │              44 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 5,720 (22.34 KB)
 Trainable params: 5,720 (22.34 KB)
 Non-trainable params: 0 (0.00 B)
- Model 3 (Starter_Code_Optimization_2) had 3 hidden layers, each with 35 neurons (the number of features the model was fed). For this model the number of neurons in each layer decreased because the "IMCOME_AMT" was convered from a categorical variable to a numerical variable. 
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 35)                  │           1,260 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 35)                  │           1,260 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 35)                  │           1,260 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │              36 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,816 (14.91 KB)
 Trainable params: 3,816 (14.91 KB)
 Non-trainable params: 0 (0.00 B)
- Model 4 (Starter_Code_Optimization_3) had 3 hidden layers, each with 87 neurons (the number of features the model was fed). For this model the number of neurons in each layer increased from the previous model because the NAME variable was also used as a feature which was a categorical variable.
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 87)                  │           7,656 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 87)                  │           7,656 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 87)                  │           7,656 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │              88 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 23,056 (90.06 KB)
 Trainable params: 23,056 (90.06 KB)
 Non-trainable params: 0 (0.00 B)
 
Were you able to achieve the target model performance?
- Yes model 4 achieved the target model performance, accuracy above 0.75. 

What steps did you take in your attempts to increase model performance?
- The first attempt to increase model performace was to add a 3rd hidden layer.  The model did not seem to change in response to that so the next step was to do some feature engineeering.  In this step the "INCOME_AMT" variable was converted from categories (intervals) to a single numerical value. In the feature engineering step the "NAME" variable was also kept and used a feature, the only change that was made was that insitutions with a count < 50 were grouped into an "OTHER" category.  

# Summary: 
Overall this analysis showed that when creating a model to solve a classification problem it is often not enough to add or change the hidden layers of a model but often times it is necessary to do some feature engineering and change the way different features are used in the model.  
