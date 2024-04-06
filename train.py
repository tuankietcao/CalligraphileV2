import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
import keras

def train_model(sample_per_letter, epochs):
  data_path = 'Images/Images'
  data = []
  folders = os.listdir(data_path)

  for folder in folders:
      files = os.listdir(os.path.join(data_path, folder))
      count = 0
      for f in files:
          if count >= sample_per_letter:  #giảm để cho nhanh
              break
          if f.endswith('.png'):
              img = Image.open(os.path.join(data_path, folder, f))
              img = img.resize((32, 32))
              img = img.convert('L') #grayscale
              img = np.asarray(img)
              data.append([img, folder])
          count += 1

  train_data, val_data = train_test_split(data, test_size=0.2)

  train_X = []
  train_Y = []
  for features, label in train_data:
      train_X.append(features)
      train_Y.append(label)

  val_X = []
  val_Y = []
  for features, label in val_data:
      val_X.append(features)
      val_Y.append(label)

  LB = LabelBinarizer() #chuyển a b c thành binary cho dễ train
  train_Y = LB.fit_transform(train_Y)
  val_Y = LB.fit_transform(val_Y)

  train_X = np.array(train_X) / 255.0
  train_X = train_X.reshape(-1, 32, 32, 1)
  train_Y = np.array(train_Y)

  val_X = np.array(val_X) / 255.0
  val_X = val_X.reshape(-1, 32, 32, 1)
  val_Y = np.array(val_Y)

  model = Sequential()
  model.add(Input(shape=train_X.shape[1:]))

  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(len(set(folders)), activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
  history = model.fit(train_X, train_Y, epochs=epochs, verbose=1)
  model.summary()
  model.evaluate(train_X, train_Y)

 # plt.figure(figsize=(8,4))

  # plt.subplot(1,2,1)
  # plt.title('Loss')
  # plt.xlabel('Epochs')
  # plt.plot(history.history['loss'])

  # plt.subplot(1,2,2)
  # plt.title('Accuracy')
  # plt.xlabel('Epochs')
  # plt.plot(history.history['accuracy']) 

  # plt.show()
  model_dir = "saved_models"
  os.makedirs(model_dir, exist_ok=True)
  model.save(os.path.join(model_dir, "my_model.keras"))
  return model, history

# Function to load the trained model from a file (optional)
def load_model(model_path):
  try:
    return keras.models.load_model(model_path)
  except FileNotFoundError:
    st.error("Trained model not found. Please train a model first.")
    return None
  