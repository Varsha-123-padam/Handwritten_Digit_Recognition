import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model('mnist_digit_recognition.h5')

# Load test data
data = np.load('mnist_data.npz')
x_test, y_test = data['x_test'], data['y_test']

# Make a prediction
index = 5  # Change this index to test different samples
sample_image = x_test[index].reshape(1, 28, 28, 1)
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)

# Display results
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {predicted_label}, Actual Label: {y_test[index]}")
plt.show()
