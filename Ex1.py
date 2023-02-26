#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello world")


# In[3]:


print("hello world")


# In[2]:


import pandas as pd


# In[10]:


dataframe=pd.read_csv('C:\\Users\\Vinod A\\Downloads\\collegePlace.csv')


# In[11]:


dataframe


# In[12]:


dataframe.tail()


# In[6]:


dataframe['Stream'].unique()


# In[24]:


dataframe


# In[13]:


import numpy as np


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


from sklearn import metrics


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


dataframe.columns


# In[20]:


feature_cols=['Age', 'Internships', 'CGPA', 'Hostel',
       'HistoryOfBacklogs', 'PlacedOrNot']


# In[21]:


x=dataframe[feature_cols]# features


# In[22]:


y=dataframe.PlacedOrNot # Target variable


# In[23]:


x.shape


# In[24]:


#x=dataframe.iloc[:,-1]


# In[25]:


#y=dataframe.iloc[:,-1]


# In[33]:


x


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[35]:


x_train.shape


# In[36]:


x_test.shape


# In[37]:


clf=DecisionTreeClassifier()


# In[38]:


clf=clf.fit(x_train,y_train)


# In[39]:



y_pred=clf.predict(x_test)


# In[40]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[42]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[1]:


x_test


# In[2]:


y_pred


# In[3]:


import numpy as np


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


from sklearn import metrics


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


dataframe.columns


# In[14]:


import numpy as np


# In[15]:


feature_cols=['Age', 'Internships', 'CGPA', 'Hostel',
       'HistoryOfBacklogs', 'PlacedOrNot']


# In[16]:


x=dataframe[feature_cols]# features


# In[17]:


dataframe


# In[1]:


dataframe


# In[21]:


# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('C:\\Users\\Vinod A\\Downloads\\mca_data.csv')

# Split data into input and output variables
X = df.drop(['Campus_placement'], axis=1)
y = df['Campus_placement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit decision tree model to training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Get input from user
tenth = float(input('Enter SSC percentage: '))
twelth = float(input('Enter HSC percentage: '))
UG = float(input('Enter UG degree percentage: '))
PG = float(input('Enter Post graduation percentage: '))
Gender = input('Enter gender (M/F): ')

UG_Course = input('Enter UG specialization 1-BCA  2-BCS  3-B.Com: ')

# Create input dataframe
input_df = pd.DataFrame({'tenth': [tenth], 'twelth': [twelth], 'UG': [UG],
                         'PG': [PG], 'Gender': [Gender], 
                         'UG_Course': [UG_Course]})


# Encode categorical variables
input_df['Gender'] = input_df['Gender'].map({'M': 1, 'F': 0})
input_df = pd.get_dummies(input_df, columns=['UG_Course'])

# Make prediction on input data
prediction = model.predict(input_df)

# Print prediction
if prediction == 1:
    print('You will get placed!')
else:
    print('Sorry, you will not get placed.')


# In[9]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('C:\\Users\\Vinod A\\Downloads\\mca_data.csv')

# Split data into input and output variables
X = df.drop(['Campus_placement'], axis=1)
y = df['Campus_placement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit decision tree model to training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Get input from user
tenth = float(input('Enter SSC percentage: '))
twelth = float(input('Enter HSC percentage: '))
UG = float(input('Enter UG degree percentage: '))
PG = float(input('Enter Post graduation percentage: '))
Gender = (input('Enter gender (M/F): '))

UG_Course = (input('Enter UG specialization 1-BCA  2-BCS  3-B.Com: '))

# Create input dataframe
input_df = pd.DataFrame({'tenth': [tenth], 'twelth': [twelth], 'UG': [UG],
                         'PG': [PG], 'Gender': [Gender], 
                         'UG_Course': [UG_Course]})


# Encode categorical variables
input_df['Gender'] = input_df['Gender'].map({'M': 1, 'F': 0})
input_df = pd.get_dummies(input_df, columns=['UG_Course'])

# Make prediction on input data
prediction = model.predict(input_df)

# Print prediction
if prediction == 1:
    print('You will get placed!')
else:
    print('Sorry, you will not get placed.')

    


# In[29]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('C:\\Users\\Vinod A\\Downloads\\mca_data.csv')

# Split data into input and output variables
X = df.drop(['Campus_placement'], axis=1)
y = df['Campus_placement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit decision tree model to training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Get input from user
tenth = float(input('Enter SSC percentage: '))
twelth = float(input('Enter HSC percentage: '))
UG = float(input('Enter UG degree percentage: '))
PG = float(input('Enter Post graduation percentage: '))
Gender =  (input('Enter gender (M/F): 0-female 1-male '))

UG_Course = (input('Enter UG specialization 1-BCA  2-BCS  3-B.Com: '))

# Create input dataframe
input_df = pd.DataFrame({'tenth': [tenth], 'twelth': [twelth], 'UG': [UG],
                         'PG': [PG], 'Gender': [Gender], 
                         'UG_Course': [UG_Course]})



# One-hot encode categorical variables

#input_df['Gender'] = input_df['Gender'].map({'M': 1, 'F': 0})
#input_df['UG_Course'] = input_df['UG_Course'].map({'1': 1, '2': 2,'3': 3})

#input_df = pd.get_dummies(input_df, columns=['Gender'])

# Reorder columns to match training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Make prediction on input data
prediction = model.predict(input_df)

# Print prediction
if prediction == 1:
    print('You will get placed!')
else:
    print('Sorry, you will not get placed.')


# In[ ]:





# In[ ]:




