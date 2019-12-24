# ./python_code/api.py
import os
import pickle
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import numpy as np
import pandas as pd
import pdb

app = Flask(__name__)
CORS(app)
api = Api(app)

# Require a parser to parse our POST request.
# TODO: can we automatically parse the args from the cmd, instead of having to update this everytime?
parser = reqparse.RequestParser()
parser.add_argument("LOAN_AMOUNT")
parser.add_argument("LENDER_TERM")
parser.add_argument("NUM_BORROWERS")
parser.add_argument("PERCENT_FEMALE")
parser.add_argument("PLANNED_DURATION")
parser.add_argument("CURRENCY_POLICY")
parser.add_argument("REPAYMENT_INTERVAL")
parser.add_argument("DISTRIBUTION_MODEL")

# Unpickle our model so we can use it!
if os.path.isfile("./low_categorical_feat_model.pkl"):
  model = pickle.load(open("./low_categorical_feat_model.pkl", "rb"))
else:
  raise FileNotFoundError

class Predict(Resource):
  def post(self):
    args = parser.parse_args()
# Sklearn is VERY PICKY on how you put your values in...
    X = [
          args["LOAN_AMOUNT"],
          args["LENDER_TERM"],
          args["NUM_BORROWERS"],
          args["PERCENT_FEMALE"],
          args["PLANNED_DURATION"],
          args["CURRENCY_POLICY"],
          args["REPAYMENT_INTERVAL"],
          args["DISTRIBUTION_MODEL"]
    ]
    col_names = ["LOAN_AMOUNT", "LENDER_TERM", "NUM_BORROWERS", "PERCENT_FEMALE", 
                "PLANNED_DURATION", "CURRENCY_POLICY", "REPAYMENT_INTERVAL", "DISTRIBUTION_MODEL"]
    df_test = pd.DataFrame(data = [X], columns = col_names)
    df_test[['LOAN_AMOUNT', 'LENDER_TERM', 'NUM_BORROWERS', 'PERCENT_FEMALE', 'PLANNED_DURATION']] = df_test[['LOAN_AMOUNT', 'LENDER_TERM', 'NUM_BORROWERS', 'PERCENT_FEMALE', 'PLANNED_DURATION']].astype(float)
    df_test[['CURRENCY_POLICY', 'REPAYMENT_INTERVAL', 'DISTRIBUTION_MODEL']] = df_test[['CURRENCY_POLICY', 'REPAYMENT_INTERVAL', 'DISTRIBUTION_MODEL']].astype('category')
    _y = model.predict(df_test)[0]
    return {"class": _y}
api.add_resource(Predict, "/predict")

if __name__ == "__main__":
  app.run(debug=True)