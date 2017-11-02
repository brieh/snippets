#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from numpy.linalg import inv

#Code fragments provided by Raj Patel

# this function standardizes a column
# for each value in the column, we subtract the mean and divide by the std dev
def standardize(col):
    # do not standardize the Boolean columns
    if any(col.name in x for x in discr_cols.values()):
        return (col)
    else:
        return ((col-col.mean())/col.std())

# this function takes as input a weight vector, w, a validation set, df_val,
# and a target vector,v, and outputs the squared-error loss 
def calc_val_err(w, df_val, t):
    h = df_val.dot(w)
    return sum((t-h)**2)

# this function takes as input a regularization parameter, l, a design 
# matrix, dm, & a target vector, y, and outputs the optimal weight vector, w
def calc_weights(l, dm, y):
    D = len(dm.columns)
    w = inv(np.add((l*np.identity(D)),(dm.transpose().dot(dm)))).\
            dot(dm.transpose()).dot(y)
    return (w)
      
# a list of the labels for the discrete columns
discr_cols = {'country_group': ['CAN','EURO','USA'],
              'Position':['C','R','D','L']}



#Load your dataset here
input = r'preprocessed_datasets.csv'
df = pd.read_csv(input)

#Create your Features "x" and Target Values "targets" here from dataset
#preprocessing steps, standardize features

# separate off our target vector, the sum_7yr_GP column, then drop it from 
# the data matrix
targets = df['sum_7yr_GP']
df = df.drop('sum_7yr_GP', axis=1)

# separate off the DraftYear column to use later for separating out training 
# and validation set
DraftYear = df['DraftYear']

# as instructed, drop 'id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'country',
# 'Overall', 'GP > 0'
df = df.drop('id', axis=1)
df = df.drop('PlayerName', axis=1)
df = df.drop('sum_7yr_TOI', axis=1)
df = df.drop('DraftYear', axis =1)
df = df.drop('Country', axis=1)
df = df.drop('Overall', axis=1)
df = df.drop('GP_greater_than_0', axis=1)



# replace discrete columns by dummy Boolean columns
for i in discr_cols:
    for j in discr_cols[i]:
        df[j] = pd.Series(np.where(df[i] == j, 1, 0), index=df.index)
    df = df.drop(i, axis=1)



# add interaction columns
# D is the number of columns before we add the interaction columns
# As I understand it, we don't add interactions among boolean columns in the
# same group because they will have 0 standard deviation. I add them here bc
# later I check for columns with 0 standard deviation so they will be dropped
# at that point
D = len(df.keys())
for i in range(D):
    j = i+1
    while (j < D):
        df[df.keys()[i] + "x" + df.keys()[j]] = df.iloc[:,i]*df.iloc[:,j]
        j = j+1



# here I standardize the columns
# however first I check to make sure the stddev isn't zero
for col in df:
    # if stddev == 0, the column is dropped
    # this will drop all the columns representing interactions between the 
    # dummy columns within a group, and possible others
    if df[col].std() != 0:
        df[col] = standardize(df[col])
    else:
        df = df.drop(col, axis=1)



# add a bias column
df['bias'] = 1.0



# separate out my training set
dft = df.loc[DraftYear.isin(['2004','2005','2006'])]
targets_t = targets.loc[DraftYear.isin(['2004','2005','2006'])]

# separate out my testing set
dftest = df[DraftYear == 2007]
targets_test = targets[DraftYear == 2007]



#Use the following values of lambda
lambdas =[0,0.01,0.1,1,10,100,1000]

# I store my squared error values in a dataframe called results that looks like
# -------------------------------------------------
# |        | 0   0.01    0.1   1   10   100   1000|
# ------------------------------------------------|
# |   cvErr|                                      |
# ------------------------------------------------|
# |testErr|                                      |
# ------------------------------------------------|
results = pd.DataFrame(0.0, columns=lambdas, index=['cvErr','testErr'])


for l in lambdas:
    # first the CV

    # number of rows used for validation will be total number of rows in the 
    # entire training set, divided by ten
    # use the remaining rows for training 

    # first validation set will be the first 63 rows
    # kth validation set will start at row 63*(k-1)
    N_TRAIN = int(len(dft.index)/10)
    first_row = 0

    # total_err sums the error for the k folds 
    total_err = 0

    for i in range(10):
        last_row = first_row+N_TRAIN

        ## Testing Features
        x_test = dft[first_row:last_row]
        ## Training Features 
        x_train = dft.drop(dft.index[first_row:last_row])

        ## Testing values
        t_test = targets_t[first_row:last_row]
        ## Training values 
        t_train = targets_t.drop(targets_t.index[first_row:last_row])

        # fit model using the training portion to get a w
        w = calc_weights(l, x_train, t_train)
        err = calc_val_err(w, x_test, t_test)

        total_err = total_err + err

        first_row = first_row+N_TRAIN

    avg_err = total_err/10.0
    results[l]['cvErr'] = avg_err


    # now find the error just training on DraftYears 2004-2006
    # and testing on DraftYear 2007 
    w = calc_weights(l, dft, targets_t)
    err = calc_val_err(w, dftest, targets_test)
    results[l]['testErr'] = err



# plot for the crossvalidation
#Find the best value of Error from cross validation error set

cv_best_lambda = results.loc['cvErr'].idxmin()
cv_best_val_error = results.loc['cvErr'].min()

cv_lmb = "Best Lambda: "+ str(cv_best_lambda)
cv_error = "Error at Best Lambda: %.4f"%cv_best_val_error

# test set plot
# Find the best value of Error from test set error

test_best_lambda = results.loc['testErr'].idxmin()
test_best_val_error = results.loc['testErr'].min()

lmb = "Best Lambda: "+ str(test_best_lambda)
error = "Error at Best Lambda: %.4f"%test_best_val_error

# Produce a plot of results.
#Change the details below as per your need
plt.semilogx(lambdas[1:], list(results.loc['cvErr'])[1:], \
    label='cross validation error')
plt.semilogx(cv_best_lambda,results.loc['cvErr'].min(), marker='o', color='r',\
    label="Best Lambda")
plt.semilogx(lambdas[1:], list(results.loc['testErr'])[1:], \
    label='test set (DraftYear = 2007) error')
plt.semilogx(test_best_lambda,results.loc['testErr']. min(), marker='o',\
    color='r',label="Best Lambda")
plt.ylabel('Sum Squared Error')
plt.text(5, 116, cv_lmb, fontsize=15)
plt.text(5, 109, cv_error, fontsize=15)
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.xlabel('Lambda')
plt.savefig('plots.pdf')
plt.show()


# inspect learned weight magnitudes given by best value of lambda
w = calc_weights(cv_best_lambda, dft, targets_t)
# create a dataframe of the sorted weights with the column names attached
weights = pd.Series(w, index=dft.columns).abs().sort_values(ascending=False)
#print the highest weights to the screen
print (weights.head(10))

plt.plot(w) 
plt.ylabel('weight')
plt.xlabel('column')
plt.savefig('weights.pdf')
