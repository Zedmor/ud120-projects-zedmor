#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    for x in range(0,len(predictions)):
        #  print(abs(predictions[x]-net_worths[x])/abs(net_worths[x]))
        if abs(predictions[x]-net_worths[x])/abs(net_worths[x])<0.1:
            cleaned_data.append([ages[x], net_worths[x], abs(predictions[x]-net_worths[x])/abs(net_worths[x])])
            print([ages[x], net_worths[x], abs(predictions[x]-net_worths[x])/abs(net_worths[x])])

    ### your code goes here

    print len(cleaned_data)
    return cleaned_data

