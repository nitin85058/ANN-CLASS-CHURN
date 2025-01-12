import streamlit as slt
import numpy as numoy
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)


with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
slt.title('CUSTOMER CHURN PREDICTION')

geography=slt.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=slt.selectbox('Gender',label_encoder_gender.classes_)
age=slt.slider('Age',18,92)
balance = slt.number_input('Balance')
credit_score = slt.number_input('Credit Score')
estimated_salary = slt.number_input('Estimated Salary')
tenure = slt.slider('Tenure', 0, 10)
num_of_products = slt.slider('Number of Products', 1, 4)
has_cr_card = slt.selectbox('Has Credit Card', [0, 1])
is_active_member = slt.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_probab=prediction[0][0]

slt.write(f'Churn Probability: {prediction_probab:.2f}')

if prediction_probab>0.5:
    slt.write('The customer is likely to churn')
else:
    slt.write('The customer is not likely to churn')