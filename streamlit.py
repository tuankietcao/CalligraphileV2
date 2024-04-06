import streamlit as st
import numpy as np
import keras
from streamlit_drawable_canvas import st_canvas
from train import train_model, load_model
from keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt
from time import time


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
  st.image(r'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARMAAAC3CAMAAAAGjUrGAAAAb1BMVEX///8AAACqqqp7e3v//f75+fn19fX6+vp/f388PDzo6Og4ODhBQUHt7e3c3NwiIiIzMzNWVlYWFhZdXV3FxcWWlpawsLBlZWW4uLhqamqJiYkdHR1PT09JSUmgoKDMzMwtLS0MDAxycnKRkZHi4eHX5JImAAADpklEQVR4nO2c23KCMBRFk1YQwQvebW2rtv3/byzEgGCikheOnb3Xg3Xsy2FNkrPJBJQihBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQv4B6Wq93qXSVTwT8bc2vMXSlTwNk4W2LKbStTwJaa2kgFJK0r1uspKu5wlIZ7rNTroicRwlWr9I1yTN0lGi9Vq6KGHisUfKu3RVwnilvEpXJUYalZ/xyCNlKV2bEJPTyPyNMo+UuXBxMpRRLTPfhr6Rkg+F6xPgHNVG5sojn5QMTorNJXYx9UpBW2jbSorp41tTfiUr7B17j5OU33eT8tM3UqACrd0cMEpe9IfZNfHklG/ZKnulrUTrj3NOcaS8yZbZJ9dKtP40LcYJb4lsnT1ilZjldWOvfmz+cx3eYLZSmkpe6sv3hbeTaJ09YjvOlZJipLjhDWVzqZlLmkqKNcUstA0pKBsGt5Vo25Lr8Aaj5MbEaUqxI+VHttLecJuwT4rJKVRSS6nCG0oyifaPlNThDWWTOsrvrSUVY+ky+yTupKQKbxDEWTclQNtr9vaugxKYm+EAJSj79QFKliAzx24B3G3CaEry7koi6WL7IUDJHGuUdFlLcpATfiFKQCZO96iGo6R7E6YSKlEPlewn0sX2Q9w9qs1ATtoH5BIqcScOlVyzwFLSZXnVIEoCohrKgxgBTZhKqERRiSUgqqEoCWjCKOeQqMQlYOKgKJmbq+20vIIoGVLJNdGyu5KNdLH9QCUOQyq5xirp1HFAlNiOQyUNApR8SdfaEwFKUM7vBeQSECUhUQ1ESUguAVGizu9Cysuvj5SgnBhXv1l1vY+UoJwYVwOVWikbKjkzGBRS8gc2sJRsinFSSDk+VoLyjJJSZbspRsrEfSsS6ihRh7IHl9MnXtxXspWutD/KZdVKuTtSgJSor2paFNPnzpqCpMTmtcSsKbe7D5QSNa4WUNOSfe8wgVOiTvayX+9IgXqcraC+8Neb0+cT5JRnRaobUgonnvCGpqSMJy0pbniDU6J22pESn5q/jUGePmnQ3hxwY/4RbpQolbQnSmLWlEt4O4I8kNNirl0pl+5zBDlE38bpMsl56yCDHSVKuTfDNuYXUmaYSlJHSZ1Tcg05cZSaepzYlvyL9a7GCyufEysFlRsb9d+4RpR69zvJQI4Ce3mrNZz2+2y03b7/7HYr0MXVMv35Wq0Oh+l0gtl3CSGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhPxn/gBThyW6Ds2TWgAAAABJRU5ErkJggg==', width=100)
  st.title('Calligraphile')
  sample_per_letter = st.text_input('Sample per letter (100 - 15000):', key='sample_per_letter')
  epochs = st.text_input('Epochs:', key='epochs')
  if st.button("Train Model"):
    start = time()
    trained_model = train_model(int(sample_per_letter), int(epochs))
    st.session_state['trained_model'] = trained_model
    st.success("Model Trained Successfully!! Time: "+ str(round(time()-start,1)))
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.plot(trained_model[1].history['loss'])

    plt.subplot(1,2,2)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(trained_model[1].history['accuracy'])
    st.pyplot(plt.gcf())

def predict_page():
  st.title('Calligraphile')
  st.write('Draw a letter and the AI will guess it:')
  model = st.session_state['trained_model']
  if model is None:
    try:
      model_path = "saved_models/my_model.keras"
      model = load_model(model_path)
      st.session_state['trained_model'] = model
    except FileNotFoundError:
      st.error("Trained model not found. Please train a model first.")
      return

  canvas = st_canvas(
      stroke_width=15,
      stroke_color='rgb(255, 255, 255)',
      background_color='rgb(0, 0, 0)',
      height=400,
      width=400,
      key="canvas"
  )

  if st.button("Predict"):
      img = canvas.image_data
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, dsize=(32, 32))  # Adjust dimensions
      img = img.astype("float32") / 255.0      
      img = img_to_array(img)  # Convert canvas data to NumPy array

      img = np.expand_dims(img, axis=0)  # Add a dimension for batch processing
      # Make prediction using the trained model
      prediction = model[0].predict(img)
      class_map = { i: chr(i + 65) for i in range(26) }
      # Get top 5 predictions with probabilities (using NumPy's top_k function)
      top_k_predictions = np.argsort(prediction[0])[-5:][::-1] 
      top_k_proba = prediction[0][top_k_predictions] 

      top_labels = [class_map[i] for i in top_k_predictions]

      st.success("Predicted Classes:")
      for label, proba in zip(top_labels, top_k_proba):
          confidence = round(proba * 100, 2)
          st.write(f"- {label}: {confidence}%")
if __name__ == "__main__":
  main()