#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Plot salary and bonus of people to see what'up there
temp_dataset = data_dict
temp_data = featureFormat(temp_dataset,['salary','bonus'],remove_NaN=True)
#plt.scatter(zip(*temp_data)[0],zip(*temp_data)[1])
####plt.show()
#Now it's evident some people have salary that's too big. Who are they?
#Reorder dictionary by salary to make sure it's real people there to figure out outlier source
#get a list key-salary
salarylist=[]
for keys,values in data_dict.items():
    if values['salary']<>'NaN': salarylist.append([keys,values['salary']])
#print sorted(salarylist, key=lambda salar:salar[1],reverse=True)
#TOTAL is input error. Delete it from dataset.
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
#ispoi,temp_data = targetFeatureSplit(featureFormat(temp_dataset,features_list,remove_NaN=True))
#class Flexlist(list):
#    def __getitem__(self, keys):
#        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
#        return [self[k] for k in keys]
#colors=Flexlist(['b','r'])
#print map(lambda i: colors[int(i)], ispoi)
#plt.scatter(zip(*temp_data)[0],zip(*temp_data)[1],color=map(lambda i: colors[int(i)], ispoi))
######plt.show()
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict



#print df.index,df.ix
#df.rename(columns = {0: 'Name'}, inplace = True)

#print df.divide(df.from_poi_to_this_person,df.to_messages)

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

df = pd.DataFrame.from_dict(data_dict, orient='index')
df=df.convert_objects(convert_numeric=True)
df['from_poi_ratio']= df.from_poi_to_this_person / df.to_messages
df['to_poi_ratio'] = df.from_this_person_to_poi / df.from_messages
df['Name'] = df.index

features_list = ['poi','from_poi_ratio','to_poi_ratio']+finfeatures+emailfeatures
#,'salary','shared_receipt_with_poi']
#print df.from_poi_ratio

num_features = 10 # 10 for logistic regression, 8 for k-means clustering

#my_feature_list = [target_label] + best_features.keys()




#emaillist=[]
#a=0
#b=0
#for keys,values in data_dict.items():
#    if values['email_address']=='NaN': a=a+1
#    b=b+1
    #print values['email_address']
#print a,b

#my_dataset= pd.DataFrame.to_dict(df,orient='index')


#print df[['from_poi_ratio','to_poi_ratio']].values
testframe1 = df.fillna(0)# df.dropna(subset = features_list)
labels = testframe1['poi'].values.tolist()
labels = [int(elem) for elem in labels]
features = testframe1[features_list[1:]].values.tolist()
#print features , labels


k_best = SelectKBest(k=num_features)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:num_features])
print "{0} best features: {1}\n".format(num_features, k_best_features.keys())


clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

features = k_best_features.keys()

#data = featureFormat(my_dataset, features_list, sort_keys = False)
#labels, features = targetFeatureSplit(data)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
iso = manifold.TSNE(n_components=2, init='pca', random_state=0)
#t0 = time()
data_projected = iso.fit_transform(features)

from sklearn.manifold import Isomap
#iso = Isomap(n_components=2)
data_projected = iso.fit_transform(features)
print data_projected.shape
import matplotlib.pyplot as plt
xval=data_projected[:, 0]
indx=int(np.where(xval==min(xval))[0])
colormap = np.array(['g','r'])
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=colormap[labels],marker='o',s=50)
#,edgecolor='none', alpha=0.1, cmap=plt.cm.get_cmap('brg', 2));
#plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

clf.fit(data_projected,labels)


#print my_dataset.keys()[indx]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.decomposition import PCA
trans = PCA(0.99)
iso = PCA(0.95)

#print features.shape
#features=trans.fit_transform(features)
#print features.shape

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

clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for prediction, truth in zip(predictions, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
    else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break

try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                       true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."
#clf.fit(features_train,labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)