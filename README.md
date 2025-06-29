# 🖋️ Handwritten Digit Recognition Web App

A complete end-to-end project for recognizing **handwritten digits (0–9)** using a custom-trained Convolutional Neural Network (CNN), deployed via **Streamlit** for interactive predictions. The model is trained using the **MNIST dataset** and saved in HDF5 format for reuse.

---

## 🚀 Features

- 🧠 Train a CNN model on the MNIST digit dataset
- 💾 Save & load model using `.h5` (Keras HDF5) format
- 📊 Preprocess and store data using NumPy and `.npz` format
- 🌐 Launch an interactive **web app** using Streamlit
- 🖼️ Upload a handwritten digit image or draw using canvas
- 🤖 Predict single-digit classes (0–9)

---

## 🗂️ Project Structure

```shell
HandWrittenDigitRecognition/
├── app.py                      # Streamlit web app frontend
├── dataset_loader.py           # Downloads + saves MNIST dataset to .npz
├── model_builder.py            # Trains CNN model and saves to .h5
├── model_tester.py             # [Optional] Manually test predictions
├── mnist_data.npz              # Saved dataset (created by dataset_loader.py)
├── mnist_digit_recognition.h5  # Trained model (created by model_builder.py)
├── tf_env/                     # Python virtual environment (excluded in .gitignore)
└── README.md                   # You're reading it!
