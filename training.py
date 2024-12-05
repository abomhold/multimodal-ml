import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, regularizers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import pickle
# Mount Google Drive
#from google.colab import drive
#drive.mount('/content/drive', contentforce_remount=False)
# Load Data
data = pd.read_csv('text/cleaned.csv')
# print(raw_data.head())
# VARS
# input_col = raw_data['words']
output_col = data['age'].astype(int)
loss='mse'
# metrics='mae'
# loss='binary_crossentropy'
# metrics='accuracy'
embedding_dim=100
rnn_units=150

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Function to create sequences
def create_sequences(data):
    all_text = data.astype(str)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_text)
    sequences = tokenizer.texts_to_sequences(all_text)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    return tokenizer, sequences



# # Create tokenizer, sequences, and labels
# tokenizer, sequences = create_sequences(data["words"])
# pickle_out = open("/content/drive/MyDrive/555_cloud_assests/tokenizer.pickle","wb")
# pickle.dump(tokenizer, pickle_out)
# pickle_out.close()
# pickle_out = open("/content/drive/MyDrive/555_cloud_assests/sequences.pickle","wb")
# pickle.dump(sequences, pickle_out)
# pickle_out.close()

#pickle_in = open("/content/drive/MyDrive/555_cloud_assests/tokenizer.pickle","rb")
#tokenizer = pickle.load(pickle_in)
#pickle_in.close()
#pickle_in = open("/content/drive/MyDrive/555_cloud_assests/sequences.pickle","rb")
#sequences = pickle.load(pickle_in)
#pickle_in.close()


tokenizer, sequences = create_sequences(data['words'])
vocab_size = len(tokenizer.word_index) + 1


# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)


X = sequences
print(X)
y = output_col
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# Training and testing a neural network
act = ['sigmoid', 'linear', 'tanh', 'relu']
# model = Sequential([
#     layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=81),
#     layers.LSTM(units=128, return_sequences=True),
#     layers.LSTM(units=64),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=1, activation='linear')
# ])

model = Sequential([
    layers.Input(shape=(81,)),
    layers.Embedding(vocab_size, embedding_dim),
    layers.SpatialDropout1D(0.2),
    layers.GlobalAveragePooling1D(),  # Instead of Flatten
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    # layers.Dense(256, activation='linear'),
    # layers.Dropout(0.4),
    layers.Dense(128, activation='sigmoid'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='linear'),
    layers.Dropout(0.1),
    layers.Dense(1,activation='relu')
])


# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
# print(y_pred.shape)
# y_pred = y_pred[:,-1,0]
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))