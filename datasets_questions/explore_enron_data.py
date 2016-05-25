#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


#print(len(enron_data))
#print (enron_data['SKILLING JEFFREY K']['total_payments']);
#print (enron_data['LAY KENNETH L']['total_payments']);
#print (enron_data['FASTOW ANDREW S']['total_payments']);
names = enron_data.keys()
counter = 0
counter2  = 0
for x in range(0, len(names)):
    if enron_data[names[x]]["total_payments"]=='NaN' and enron_data[names[x]]["poi"]==True: counter +=1;
    #if enron_data[names[x]]["email_address"]=='NaN': counter2 +=1;
print(counter)
#print(146-counter2)

