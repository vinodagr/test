

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('mca_data.csv')

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
# Add a text input widget
tenth = st.text_input('Enter your 10th percentage')


twelth = st.text_input('Enter your 12th percentage')

UG = st.text_input('Enter your UG percentage')
PG = st.text_input('Enter your PG percentage')
Gender = st.selectbox('Select your gender', ['Male', 'Female'])

UG_Course = st.text_input('Enter UG specialization 1-BCA  2-BCS  3-B.Com: ')

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
#prediction = model.predict(input_df)

    # Add button to make prediction
        submit_button = st.form_submit_button(label='Predict Placement')

    # Display prediction on button click
    if submit_button:
        prediction =  model.predict(input_df)
        st.write(prediction)
# Print prediction
#if prediction == 1:
 #   st.write('You will get placed!')
#else:
 #   st.write('Sorry, you will not get placed.')




