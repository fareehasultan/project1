import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = joblib.load("livemodelv1.pkl")

data = pd.read_csv('mobile_price_range_data.csv')
x = data.iloc[:,:-1]
y = data.iloc[:, -1]             ]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#make predictions for X_test set
y_pred = model.predict(X_test)

st.title("model accuracy and real time prediction")

st.write(f"Model {accuracy}")

st.header("Real-Time Prediction")
input_data = []
for col in X_test.columns:
    input_value = st.numner_input(f'Input for feature {col}' , value='0.0')
    input_data.append(input_value)
    #convert input data to dataframe
    input_df = pd.DataFrame([input_data], columns=X_test.columns)
    #make predictions
    if st.button("Predict")
    prediction = model.predict(input_df)
    st.write(f'P[rediction: {prediction[0]}')
