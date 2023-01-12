#install all the packages that are listed below for smooth running of code
#pip install pandas
#pip install numpy
#pip install -U scikit-learn
#pip install tensorflow


import requests
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, Flatten, Dot, Add, Concatenate, Dense
from keras.models import Model

#Importing data from link:
 
url='https://www.imdb.com/search/title/?groups=top_250&sort=user_rating'
data = requests.get(url)

with open ('movie_ratings.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)



#Data collection:

# Load data from a CSV file
data = pd.read_csv("movie_ratings.csv")

users=[]
movies=[]
#Data Processing:

# Remove any rows with missing data
data = data.dropna()

# Normalize ratings to be between 0 and 1
data["rating"] = data["rating"] / 5

# Extract relevant features from the movies
data["year"] = data["date"].apply(lambda x: x.year)
data["genre"] = data["genres"].apply(lambda x: x.split(",")[0])



#Model training:

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Extract the user and movie IDs
user_ids = data["user_id"].values
movie_ids = data["movie_id"].values

# Create an embedding layer for the user IDs
user_id_input = Input(shape=[1], name="user_id_input")
user_id_embedding = Embedding(input_dim=len(users) + 1, output_dim=1, name="user_id_embedding")(user_id_input)

# Create an embedding layer for the movie IDs
movie_id_input = Input(shape=[1], name="movie_id_input")
movie_id_embedding = Embedding(input_dim=len(movies) + 1, output_dim=1, name="movie_id_embedding")(movie_id_input)

# Flatten the embeddings
user_id_embedding = Flatten()(user_id_embedding)
movie_id_embedding = Flatten()(movie_id_embedding)

# Combine the user and movie embeddings into a single vector
input_vectors = Concatenate()([user_id_embedding, movie_id_embedding])

# Add a dense layer to the model
x = Dense(64, activation='relu')(input_vectors)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# Output layer
output = Dense(1)(x)

# Create the model
model = Model(inputs=[user_id_input, movie_id_input], outputs=output)

# Compile the model
model.compile(loss='mse', optimizer='adamax')

# Fit the model to the training data
history = model.fit([train_data["user_id"], train_data["movie_id"]], train_data["rating"], epochs=50, verbose=1)



#Model Evaluation:

# Make predictions on the test data
predictions = model.predict([test_data["user_id"], test_data["movie_id"]])
predictions = predictions.flatten()

# Calculate the mean squared error of the predictions
mse = mean_squared_error(test_data["rating"], predictions)
print("Mean Squared Error: ", mse)
