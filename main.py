import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')


@app.route('/predict', methods=['POST'])
def predict():
        file = request.files.get('file')
        filename = secure_filename(file.filename)
        file.save(filename)
        img = image.load_img(filename, target_size=(224, 224))
        os.remove(filename)

        img = img.resize((224, 224))  # resize the image to match the input size of the model
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add a batch dimension to the input

        # Make the prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        # return the predicted label
        if predicted_class == 0:
            return jsonify({'label': 'Bed'})
        elif predicted_class == 1:
            return jsonify({'label': 'Chair'})
        else:
            return jsonify({'label': 'Sofa'})

if __name__ == '__main__':
    app.run()









# from flask import Flask, jsonify, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import matplotlib.pyplot as plt
# # import cv2
#
# # Define the Flask app
# app = Flask(__name__)
#
# # Load the trained model
# model = tf.keras.models.load_model('model.h5')
#
# # Define the class labels
# class_labels = ['Bed', 'Chair', 'Sofa']
#
#
# # model = load_model('model.h5')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the image from the request
#     # image = cv2.imread('/path_to_image/messi5.jpg', 0)
#     image = plt.imread('Avalon_Sofa.jpg')
#     # image = request.files('./Avalon_Sofa.jpg').read()
#     image = Image.open(io.BytesIO(image))
#     image = image.resize((150, 150)) # Resize the image to the same size used for training
#     image = np.array(image) / 255.0 # Normalize the image data
#
#     # Make the prediction
#     prediction = model.predict(np.expand_dims(image, axis=0))[0]
#     predicted_class = class_labels[np.argmax(prediction)]
#
#     # Return the predicted class as JSON
#     response = {'predicted_class': predicted_class}
#     return jsonify(response)
#
# if __name__ == '__main__':
#     app.run(debug=True)