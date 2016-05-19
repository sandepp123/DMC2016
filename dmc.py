import pandas as pd
import numpy as np
import Model
import matplotlib.pyplot as plt


print "Reading Train"
train=Model.file_open('orders_train.txt',';')
print "Readin Train Finished"
print "reading Testing"
test=Model.file_open('orders_class.txt',';')
print "reading Testing Finished"
##########cleaning data##########
Missing_value_columns=['productGroup','rrp','voucherID']
train=Model.data_clean(train,Missing_value_columns)
test=Model.data_clean(test,Missing_value_columns)

##########################converting product group ,colo and device ID into object format########
train = Model.converting_to_object(train,['colorCode','productGroup','deviceID'])
test  = Model.converting_to_object(test,['colorCode','productGroup','deviceID'])

###########################removing low frequent customer voucer and article ID###########################3
#train=Model.Normalizing(train,['articleID','voucherID'],[200,20])  too much time required for conversion not very efficient for accuracy
#test = Model.Normalizing(test,['articleID','voucherID'],[200,20])
########################################################labeling#################
categorica_variable=list(train.dtypes.loc[train.dtypes=='object'].index)
train,test=Model.labeling(train,test,categorica_variable[2:])
print train.dtypes
print test.dtypes

##################spliting dataset for testing and training
train_train,test_train=Model.split_dataset(train)

##################classification######################################
independent=['articleID','colorCode','sizeCode','productGroup','quantity','price','rrp','voucherID', 'voucherAmount','customerID',
			'paymentMethod']
dependent=['returnQuantity']
prediction=Model.classification(train_train,test_train,independent,dependent)
