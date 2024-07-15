from flask import Flask, request, jsonify
import requests
from transformers import DistilBertTokenizer
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
import numpy as np
import os

app = Flask(__name__)

deployed_model_name = "itsmtickets"
rest_url = "http://modelmesh-serving:8008"
infer_url = f"{rest_url}/v2/models/{deployed_model_name}/infer"

tokenizer = DistilBertTokenizer.from_pretrained('alezzandro/itsm_tickets')

id2label = {0: "WebServer", 1: "Database", 2: "Filesystem"}

@app.route('/')
def hello():
    return "Hello, you should call /predict_text with a POST request!"

def ml_rest_request(text):
    encoded_input = tokenizer(text, return_tensors='tf')
    json_data = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": encoded_input['input_ids'].shape.as_list(),
                "datatype": "INT64",
                "data": encoded_input['input_ids'].tolist()
            },
            {
                "name": "attention_mask",
                "shape": encoded_input['attention_mask'].shape.as_list(),
                "datatype": "INT64",
                "data": encoded_input['attention_mask'].tolist()
            }
        ]
    }

    response = requests.post(infer_url, json=json_data)
    response_dict = response.json()
    return response_dict['outputs'][0]['data']

def get_max_value_position(arr):
    max_value = max(arr)
    max_index = arr.index(max_value)
    return max_index

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        # 1. Get Text from the POST Request
        data = request.get_json()
        text_to_predict = data.get('text')

        if not text_to_predict:
            return jsonify({"error": "No 'text' field provided"}), 400
        
        # 2. Predict the Text
        prediction = ml_rest_request(text_to_predict)
        category = id2label[get_max_value_position(prediction)]

        # 3. Construct and Send the Response
        response = {
            "original_text": text_to_predict,
            "category": category
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')