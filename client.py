import requests
import json
import base64
import cv2
import pprint
# PyTorch_REST_API_URL = 'http://172.16.23.8:4003/xuni/inference'
PyTorch_REST_API_URL = 'http://10.11.33.205:4003/xuni/inference'

def predict_result(image_path):
    #image = open(image_path, 'rb').read()
    image = cv2.imread(image_path)
    image = cv2.imencode('.jpg', image)
    #str_encode = base64.b64encode(image[1]).decode()
    #payload = {'action':'inference', 'image': str_encode}
    payload = {"action":"inference", "image": image[1].tolist()}
    payload = json.dumps(payload)
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    print(r)
    
def init():
    payload = {'action':'initialize net','thisid':'0'}
    payload = json.dumps(payload)
    print(playload)
    headers = {"Content-type": "application/json"}
    r = requests.post(PyTorch_REST_API_URL, headers=headers, data=payload).json()
    print(r)


if __name__ == "__main__":
    init()
    for i in range(2):
        predict_result('data/images/RGB17.jpg')