from pandas import DataFrame, read_csv

import pandas as pd
import numpy as np
from math import e, sqrt, pi

input = r'preprocessed_datasets.csv'

df_full = pd.read_csv(input)

# as instructed, drop 'id' and 'PlayerName' columns because they are identifiers
# also drop 'Country' because it contains too many discrete values
# drop 'sum_yr' columns because they are all zero for GP>0 = no and create a divide by zero issue in the gaussian 
df_full = df_full.drop('id', axis=1)
df_full = df_full.drop('PlayerName', axis=1)
df_full = df_full.drop('Country', axis=1)
df_full = df_full.drop('sum_7yr_GP', axis=1)
df_full = df_full.drop('sum_7yr_TOI', axis=1)

# separate out my training set
dftrain = df_full.loc[df_full['DraftYear'].isin(['2004','2005','2006'])]
# now drop DraftYear column because it's how we separate training data from test data
# and if we then use it to classify, it would skew the results
dftrain = dftrain.drop('DraftYear', axis=1)

# within the training set, I further separate those with class true
# and those with class false
dftrain_y = dftrain[dftrain.GP_greater_than_0 == 'yes']
dftrain_n = dftrain[dftrain.GP_greater_than_0 == 'no']

# separate out my testing set
dftest = df_full[df_full.DraftYear == 2007]
# drop 'DraftYear' from the test set too
dftest = dftest.drop('DraftYear', axis=1)

# add a column to the test set to store my class predictions
dftest['Class_prediction'] = pd.Series('',index=dftest.index)
dftest['Accurate'] = pd.Series('',index=dftest.index)

# Dictionary to hold the probability values for the discrete columns
discr_dict = {'country_group':{'EURO':{'py':0.0, 'pn':0.0},
                               'USA':{'py':0.0, 'pn':0.0},
                               'CAN':{'py':0.0, 'pn':0.0}
                               },
              'Position':{'C':{'py':0.0, 'pn':0.0},
                          'D':{'py':0.0, 'pn':0.0},
                          'L':{'py':0.0, 'pn':0.0},
                          'R':{'py':0.0, 'pn':0.0}
                          }
              }
              
# Dictionary to hold the means and variances of the for the continuous columns
cont_dict = {'DraftAge':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'Height':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'Weight':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'Overall':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'CSS_rank':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_GP':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_G':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_A':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_P':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_PIM':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'rs_PlusMinus':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'po_GP':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'po_G':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'po_A':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'po_P':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0},
             'po_PIM':{'meany':0.0,'vary':0.0,'meann':0.0,'varn':0.0}
             }

def gaussian_p(mean,var,x):
    return (1/(sqrt(2*pi*var)))*e**(-0.5*float(x-mean)**2/var)
    
def predict_class(row):
    total_py = total_pn = 1.0
    # this is gonna go through ea element of the row and apply get_prob
    for col in row.keys():
        x = row[col]
        if col in cont_dict.keys():
            mean_y = cont_dict[col]['meany']
            var_y  = cont_dict[col]['vary']
            mean_n = cont_dict[col]['meann']
            var_n  = cont_dict[col]['varn']
            
            py = gaussian_p(mean_y,var_y,x)
            pn = gaussian_p(mean_n,var_n,x)
        elif col in discr_dict.keys():
            py = discr_dict[col][x]['py']
            pn = discr_dict[col][x]['pn']
        else:
            py = pn = 1.0
            
        total_py = total_py*py
        total_pn = total_pn*pn
    
    row['Class_prediction'] = np.where(total_py > total_pn, 'yes', 'no')
    row['Accurate'] = np.where(row['GP_greater_than_0'] == row['Class_prediction'], 1, 0) 
    
    return row
    
    



# fill in the discrete probabilities dictionary
# vals is a list of the unique values in the country_group column
vals = dftrain.country_group.unique()
for v in vals:
    Py = dftrain_y.country_group.value_counts()[v]/dftrain_y.country_group.count()
    Pn = dftrain_n.country_group.value_counts()[v]/dftrain_n.country_group.count()
    discr_dict['country_group'][v]['py'] = Py
    discr_dict['country_group'][v]['pn'] = Pn
    
# vals is a list of the unique values in the Position column
vals = dftrain.Position.unique()
for v in vals:
    Py = dftrain_y.Position.value_counts()[v]/dftrain_y.Position.count()
    Pn = dftrain_n.Position.value_counts()[v]/dftrain_n.Position.count()
    discr_dict['Position'][v]['py'] = Py
    discr_dict['Position'][v]['pn'] = Pn
    
# Next I need to fill in the continuous probabilities
for col in cont_dict:    
    cont_dict[col]['meany'] = dftrain_y[col].mean()
    cont_dict[col]['vary'] = dftrain_y[col].var()
    cont_dict[col]['meann'] = dftrain_n[col].mean()
    cont_dict[col]['varn'] = dftrain_n[col].var()
 

   
# this is gonna go through each row and apply the predict_class function 
# which will assign a class to each row and store a 'yes' or 'no' accordingly
# in the 'Class prediction' column
dftest = dftest.apply(predict_class, axis=1)

accuracy = dftest['Accurate'].sum()/dftest['Accurate'].count()

print ('accuracy is ')
print (accuracy)
    
    
    






