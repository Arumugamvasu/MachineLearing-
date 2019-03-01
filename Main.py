# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:44:49 2019

@author: Arumugam
"""

# import lib
import get_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




# data collection

data_path='oil_data_set.xlsx'
oil_data=get_data.get_data(data_path)
iol_in_data=oil_data.load_data()

#Check Not null valus 
columns=list(iol_in_data.columns.values)
len_cl=len(columns)
for i in columns:
       if int(iol_in_data[i].isnull().sum()) == 0 :
           print('{} cloumn don\'t have null vale'.format(i))
           
       else :
           
           print('{} cloumn have null vale'.format(i))


# plot monthly wise oil price
           
x_1=iol_in_data['date']

y_1=iol_in_data['oil_price_dollars']           
           
plt.plot(x_1,y_1)
plt.xlabel('month wise date')
plt.ylabel('Oil_price')
plt.show()

## data labling

#iol_in_data=lable_data.LabelEncoder(iol_in_data)

label_qulity=LabelEncoder()
iol_in_data['weather_conditions']=label_qulity.fit_transform(iol_in_data['weather_conditions'])



# seperate the dataset to respose variable and feature variable

X=iol_in_data.drop(['oil_price_dollars','date'],axis=1)          # Training fetaure 
Y=iol_in_data['oil_price_dollars'].astype('int')               # target lable for training features

#lab_enc = LabelEncoder()
#encoded = lab_enc.fit_transform(Y)
#decode=lab_enc.rfit_transform(encoded)

# Train and test data split 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

  
# Applying Stantard scaling 
# inputvalue range some tome 20 to 120 or 0.1333 to 1.23434 so we need to scalling
scaller=StandardScaler()
x_train=scaller.fit_transform(x_train)
x_test=scaller.transform(x_test)


# random forest classification initialization 
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(x_train,y_train)
pred_rfc=rfc.predict(x_test)

# lets show our model perfomance for our RFC model
print('classification_report = {}'.format(classification_report(y_test,pred_rfc)))
print('confusion_matrix = {}'.format(confusion_matrix(y_test,pred_rfc)))




print('accuracy_score = {} '.format(accuracy_score(y_test,pred_rfc)*100))


labels = [0,1]
sns.heatmap(confusion_matrix(y_test,pred_rfc), annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.show()



# testing new raw data format [Crude_oil_demant,Crude_oil_spply,weather_conditions,Monthly_Change]

x_new=[[0.62,48,1,0.024]]
x_new_fil=scaller.transform(x_new)
print(rfc.predict(x_new_fil))
