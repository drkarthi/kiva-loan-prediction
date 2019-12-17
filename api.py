# ./python_code/api.py
import os
import pickle
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
api = Api(app)

# Require a parser to parse our POST request.
parser = reqparse.RequestParser()
parser.add_argument("LOAN_AMOUNT")

# Unpickle our model so we can use it!
if os.path.isfile("./one_feat_model.pkl"):
  model = pickle.load(open("./one_feat_model.pkl", "rb"))
else:
  raise FileNotFoundError

class Predict(Resource):
  def post(self):
    args = parser.parse_args()
# Sklearn is VERY PICKY on how you put your values in...
    X = (
      np.array(
        [
          args["LOAN_AMOUNT"]
        ]
      ).astype("float").reshape(1, -1)
    )
    _y = model.predict(X)[0]
    return {"class": _y}
api.add_resource(Predict, "/predict")

if __name__ == "__main__":
  app.run(debug=True)