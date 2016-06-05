#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Plot salary and bonus of people to see what'up there
temp_dataset = data_dict
temp_data = featureFormat(temp_dataset,['salary','bonus'],remove_NaN=True)
plt.scatter(zip(*temp_data)[0],zip(*temp_data)[1])
plt.show()
#Now it's evident some people have salary that's too big. Who are they?
#Reorder dictionary by salary to make sure it's real people there to figure out outlier source
#get a list key-salary
salarylist=[]
for keys,values in data_dict.items():
    if values['salary']<>'NaN': salarylist.append([keys,values['salary']])
print sorted(salarylist, key=lambda salar:salar[1],reverse=True)
#TOTAL is input error. Delete it from dataset.
del data_dict['TOTAL']
temp_dataset = data_dict
ispoi,temp_data = targetFeatureSplit(featureFormat(temp_dataset,features_list,remove_NaN=True))
class Flexlist(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[k] for k in keys]
colors=Flexlist(['b','r'])
#print map(lambda i: colors[int(i)], ispoi)
plt.scatter(zip(*temp_data)[0],zip(*temp_data)[1],color=map(lambda i: colors[int(i)], ispoi))
plt.show()
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#let's take all features and find
#Featues that are basically useless (present in less than 30% of dataset or agreggation of.
# Useless features (rare): Loan advances, director fees
# Derivatives: Total payments, Total stock value (it's excer + restr + restr defer)
finfeatures=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
 'restricted_stock', 'director_fees']
emailfeatures=['to_messages',  'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
emailadd=['email_address']
#how many ppl have email address?


emaillist=[]
a=0
b=0
for keys,values in data_dict.items():
    if values['email_address']=='NaN': a=a+1
    b=b+1
    #print values['email_address']
print a,b

features_list=['poi']+finfeatures+emailfeatures

data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.decomposition import PCA
trans = PCA(0.99)
#iso = PCA(0.95)

#print features.shape
features=trans.fit_transform(features)
print features.shape

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)