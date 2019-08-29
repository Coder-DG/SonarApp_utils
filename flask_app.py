import os
import json

from flask import Flask, request

SAMPLES_DIR = 'samples'
app = Flask(__name__)


# curl http://127.0.0.1:5000/ -d '{"key1":"value1", "key2":"value2"}'
#   -X POST -H "Content-type: application/json"

@app.route('/', methods=['POST'])
def get_data():
    data = request.json
    tag = data.get('tag')
    if not tag:
        return {"Response": 400}
    with open(os.path.join(SAMPLES_DIR, tag), 'w') as f:
        json.dump(data, f)
    return {"Response": 200}


if __name__ == '__main__':
    if not os.path.exists(SAMPLES_DIR):
        os.mkdir(SAMPLES_DIR)
    app.run(host='0.0.0.0', port='5000')
