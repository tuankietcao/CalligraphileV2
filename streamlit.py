import streamlit as st
import numpy as np
import keras
from streamlit_drawable_canvas import st_canvas
from train import train_model, load_model
from keras.preprocessing.image import img_to_array
import cv2

if 'trained_model' not in st.session_state:
  st.session_state['trained_model'] = None

def main():
  # Define Streamlit pages with functions
  pages = {
      "Train Model": train_page,
      "Predict": predict_page
  }

  # Use Streamlit sidebar for navigation
  page = st.sidebar.selectbox("Select Page", list(pages.keys()))
  if page in pages:
      pages[page]()

def train_page():
  # Train the model and store it in session state
  sample_per_letter = st.text_input('Sample per letter (100 - 15000):', key='sample_per_letter')
  epochs = st.text_input('Epochs:', key='epochs')
  if st.button("Train Model"):
    trained_model = train_model(int(sample_per_letter), int(epochs))
    st.session_state['trained_model'] = trained_model
    st.success("Model Trained!")
    st.pyp

def predict_page():
  model = st.session_state['trained_model']  # Access model from session state

  # Load the model if not already in session state (optional)
  if model is None:
    try:
      model_path = "saved_models/my_model.keras"
      model = load_model(model_path)
      st.session_state['trained_model'] = model
    except FileNotFoundError:
      st.error("Trained model not found. Please train a model first.")
      return

  # Create a drawable canvas for user input
  canvas = st_canvas(
      stroke_width=15,
      stroke_color='rgb(255, 255, 255)',
      background_color='rgb(0, 0, 0)',
      height=150,
      width=150,
      key="canvas"
  )

  # Button to trigger prediction
  if st.button("Predict"):
      img = canvas.image_data
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, dsize=(32, 32))  # Adjust dimensions
      img = img.astype("float32") / 255.0      
      img = img_to_array(img)  # Convert canvas data to NumPy array

      img = np.expand_dims(img, axis=0)  # Add a dimension for batch processing
      # Make prediction using the trained model
      prediction = model.predict(img)
      class_map = { i: chr(i + 65) for i in range(26) }
      # Get top 5 predictions with probabilities (using NumPy's top_k function)
      top_k_predictions = np.argsort(prediction[0])[-5:][::-1]  # Get indices of top 5 classes (descending order)
      top_k_proba = prediction[0][top_k_predictions]  # Get probabilities for top 5 classes

      top_labels = [class_map[i] for i in top_k_predictions]

      st.write("Predicted Classes:")
      for label, proba in zip(top_labels, top_k_proba):
          confidence = round(proba * 100, 2)
          st.write(f"- {label}: {confidence}%")
      # ... (add class mapping or visualization if applicable)
if __name__ == "__main__":
  main()