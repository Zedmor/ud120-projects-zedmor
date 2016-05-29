#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
from sklearn import tree
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


### it's all yours from here forward!
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
#clf.fit(features,labels)
predict = clf.predict(features_test)
print clf.score(features_test,labels_test)
print metrics.precision_score(labels_test,predict)
print metrics.recall_score(labels_test,predict)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print metrics.precision_score(true_labels,predictions)

print metrics.recall_score(true_labels,predictions)

#print metrics.
#print labels_test.count(1.0)
#print len(labels_test)
#print clf.predict(features_test)
#for i in range (0,len(labels_test)):
#    if labels_test[i]==clf.predict(features_test[i])==1.0:
#        print i