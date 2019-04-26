from flask import Flask
from flask_cors import CORS, cross_origin
from util import predict_breed
from flask import Response
from flask import jsonify
from flask import request
from werkzeug import secure_filename
from json import dumps

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict_():
    '''
        returns a flower prediction
    '''
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response('No file uploaded', status=500)
        else :
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(filename)
            predicted =  predict_breed(filename)
            print(type(predicted))
            print("\n".join("{}\t{}".format(k, v) for k, v in predicted.items()))
            return dumps(predicted)
    else:
        return Response('Bad request', status=500)

if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=False, threaded=False)