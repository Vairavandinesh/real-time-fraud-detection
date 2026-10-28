import pandas as pd
from xgboost import XGBClassifier
model=XGBClassifier()
model.load_model('fraud_model.json')

transaction=pd.DataFrame([{
    "amount":9000,
    "oldbalanceOrg":10000,
    "newbalanceOrig":1000,
    "oldbalanceDest":5000,
    "newbalanceDest":14000,
    "type_CASH_IN":0,
    "type_CASH_OUT":0,
    "type_DEBIT":0,
    "type_PAYMENT":0,
    "type_TRANSFER":1
}])

prediction=model.predict(transaction)[0]

if prediction==1:
    print("Fraud transaction detected!")
else:
    print("Normal transaction")
