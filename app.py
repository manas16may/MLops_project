import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

## Load the model
regmodel=pickle.load(open('final_model.pkl','rb'))

def feature_engineering(data):
    data.replace('', float(np.nan), regex=True,inplace=True)
    data.replace(' ', float(np.nan), regex=True,inplace=True)
    data['Order_Date']=pd.to_datetime(data['Order_Date'],format="%d-%m-%Y")
    data["month"] = data.Order_Date.dt.month.astype(int)
    data["quarter"] = data.Order_Date.dt.quarter.astype(int)
    data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
    data.drop([ 'Order_Date'], axis=1, inplace=True)
    data['distance']=np.zeros(len(data.index))
    restaurant_coordinates=data[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
    delivery_location_coordinates=data[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
    data['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])
    data['distance']= data['distance'].str.extract(r'(\d+\.\d+)').astype(float)
    for x in data.columns:
        data[x]=data[x].astype("float64")
    return data



@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    labels=[x for x in request.form.keys()]
    values=[x for x in request.form.values()]
    dictt={}
    for x in request.form.keys():
        dictt[x]=request.form[x]
    data=pd.DataFrame(dictt,index=[0])
    cols_when_model_builds = regmodel.get_booster().feature_names
    final_input=feature_engineering(data)
    output=regmodel.predict(final_input[cols_when_model_builds])[0]
    return render_template("home.html",prediction_text="Time taken is {} minutes".format(output))

if __name__=="__main__":
app.run(port=5000)