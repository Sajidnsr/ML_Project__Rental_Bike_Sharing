from flask import Flask, request, render_template, url_for
import pandas as pd
from src.day_predictions import day_predictions
from src.hr_predictions import hour_predictions
from src.config import Config
from csv import DictWriter

app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home_page():
    return render_template('home.html')


@app.route('/day')
def day_prediction():
    return render_template('day.html')


@app.route('/day', methods=['POST', 'GET'])
def day_res():

    if request.method == "POST":
        season = request.form.get("season")
        mnth = request.form.get("mnth")
        holiday = request.form.get("holiday")
        weekday = request.form.get("weekday")
        workingday = request.form.get("workingday")
        weathersit = request.form.get("weathersit")
        temp = float(request.form.get("temp")) / 41
        humidity = float(request.form.get("humidity")) / 100
        windspeed = float(request.form.get("windspeed")) / 67

        day_form_data = [
            {
                'season': season,
                'mnth': mnth,
                'holiday': holiday,
                'weekday': weekday,
                'workingday': workingday,
                'weathersit': weathersit,
                'temp': temp,
                'humidity': humidity,
                'windspeed': windspeed,
            }
        ]
        headersCSV = ['season','mnth','holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'humidity', 'windspeed','casual', 'registered','count']      

        

        result = day_predictions().predict(day_form_data)
       
        # print(result)
        # print(pd.DataFrame(form_data))
        casual = int(result[0][0])
        registered = int(result[0][1])
        day_res = [
            {
                'casual': casual,
                'registered': registered,
                'count': casual + registered
            }
        ]

        z1 = day_form_data[0] | day_res[0]

        with open(r"assets\user_input\day_input.csv", "a", newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
            dictwriter_object.writerow(z1)
            f_object.close()

    return render_template('day.html', data=day_res)
    
    

@app.route('/hour')
def hour_prediction():
    return render_template('hour.html')

@app.route('/hour', methods=['POST', 'GET'])
def hour_res():

    if request.method == "POST":
        season = request.form.get("season")
        mnth = request.form.get("mnth")
        hr = request.form.get("hr")
        holiday = request.form.get("holiday")
        weekday = request.form.get("weekday")
        workingday = request.form.get("workingday")
        weathersit = request.form.get("weathersit")
        temp = float(request.form.get("temp")) / 41
        humidity = float(request.form.get("humidity")) / 100
        windspeed = float(request.form.get("windspeed")) / 67

        hour_form_data = [
            {
                'season': season,
                'mnth': mnth,
                'hr': hr,
                'holiday': holiday,
                'weekday': weekday,
                'workingday': workingday,
                'weathersit': weathersit,
                'temp': temp,
                'humidity': humidity,
                'windspeed': windspeed,
            }
        ]
        headersCSV = ['season','mnth','hr','holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'humidity', 'windspeed','casual', 'registered','count']      

        

        result = hour_predictions().predict(hour_form_data)
        
        casual = int(result[0][0])
        registered = int(result[0][1])
        hour_res = [
            {
                'casual': casual,
                'registered': registered,
                'count': casual + registered
            }
        ]


        z = hour_form_data[0] | hour_res[0]

        with open(r"assets\user_input\hour_input.csv", "a", newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
            dictwriter_object.writerow(z)
            f_object.close()

    return render_template('hour.html', data=hour_res)

if __name__ == "__main__":
    app.run(debug=False)
