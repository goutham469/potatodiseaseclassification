from flask import Flask,jsonify,request
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image




app = Flask(__name__)
CORS(app)

model = load_model("model1.keras")
print("model loading complete")


def predict_disease(image_form_user):
    print("asked to predict disease")

    image = image_form_user
    # get image here
    image = plt.imread(image)
    # print(model.summary())
    data = []
    data.append(image)
    # output = model.predict(np.argmax(np.array(data)))
    output = model.predict(np.array(data))
    classes = ['healthy','early_blight','late_blight']

    index = 0
    for i in range(len(output[0])):
        if output[0][i] == 1 :
            index = i
            break
    # print(type(output[0]),output[0],index,classes[index])
    return (classes[index])


# @app.route('/predictDisease',methods=['POST'])
# def predictDisease():

#     if request.method == 'POST' :
#         data_user = request.get_json()
#         print(data_user)
#         # the data object is to be in the format of encoded URI
#         return jsonify({"reponse":"GOOD"})

@app.route('/predictDisease', methods=['POST'])
def predict_disease():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        
        # Read the image file
        image_data = image_file.read()
        
        # Convert the image data to a PIL Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Perform any processing on the image here
        
        # Example: Convert the image to grayscale
        # image = image.convert('L')
        
        # Example: Resize the image to 256x256
        image = image.resize((256, 256))
        print(image)
        
        # Convert the resized image back to bytes
        # buffered = io.BytesIO()
        # image.save(buffered, format='JPEG')
        # resized_image_data = buffered.getvalue()
        # here print the image as 256,256,3 array
        image_array = np.array(image)
        print(image_array)
        

        data = []
        data.append(image_array)

        # output = model.predict(np.argmax(np.array(data)))
        output = model.predict(np.array(data))

        classes = ['healthy','early_blight','late_blight']

        index = 0
        for i in range(len(output[0])):
            if output[0][i] == 1 :
                index = i
                break


        # print(type(output[0]),output[0],index,classes[index])

        print(classes[index])

        print()
        # Example: Perform prediction on the resized image
        
        # Return a response (change as needed)
        return jsonify({'message': 'Image uploaded and processed successfully','predicted':classes[index]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def entry():
    return "this server is running for Plant disease prediction ML model"



if __name__ == '__main__' :
    app.run(debug=False,host='0.0.0.0')