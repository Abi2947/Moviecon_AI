#install all the packages that are listed below for smooth running of code


#pip install pandas 
#pip install numpy
#pip install -U scikit-learn
#pip install -U Flask
#pip insatll tensorflow


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify, request
import json


#Data collection:

# Load data from a CSV file
data = pd.read_csv("movie_ratings.csv")


#Data Processing:

# Remove any rows with missing data
data = data.dropna()

# Normalize ratings to be between 0 and 1
data["rating"] = data["rating"] / 5

# Extract relevant features from the movies
data['movie'] = data['movie_name'].apply(lambda x:x.movie)
data["year"] = data["date"].apply(lambda x: x.year)
data["genre"] = data["genres"].apply(lambda x: x.split(",")[0])



#Model Training:

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(train_data[["user_id", "movie_id", "year", "genre"]], train_data["rating"])


#Model Evaluation:

# Make predictions on the test data
predictions = model.predict(test_data[["user_id", "movie_id", "year", "genre"]])

# Calculate the mean squared error of the predictions
mse = mean_squared_error(test_data["rating"], predictions)
print("Mean Squared Error: ", mse)


#Deployment:

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Make a prediction using the model
    prediction = model.predict([input_data])

    # Return the prediction as a response
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
