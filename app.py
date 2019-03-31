import codecs
import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

date_parser = lambda x: pd.datetime.strptime(x, "%Y%m%d")
date_parser2 = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")

with codecs.open('Data/Power2017-2018.csv','r','utf-8-sig') as f:
    data2017 = pd.read_csv(f, encoding='utf-8', parse_dates=['日期'], date_parser=date_parser, index_col=['日期'])
    f.close()

with codecs.open('Data/Power2018-2019.csv','r','utf-8-sig') as f:
    data2018 = pd.read_csv(f, encoding='utf-8', parse_dates=['日期'], date_parser=date_parser, index_col=['日期'])
    f.close()
    
with open('Data/Taipei_temperature.csv','r') as f:
    taipei_temperature = pd.read_csv(f, encoding='utf-8', parse_dates=True, date_parser=date_parser2, index_col=0)
    f.close()
    
with open('Data/Kaohsiung_temperature.csv','r') as f:
    Kaohsiung_temperature = pd.read_csv(f, encoding='utf-8', parse_dates=True, date_parser=date_parser2, index_col=0)
    f.close()

with open('Data/Future_temperature.csv','r') as f:
    Future_temperature = pd.read_csv(f, encoding='utf-8', parse_dates=True, date_parser=date_parser2, index_col=0)
    f.close()

peaking_power2017 = data2017['2017-01-01':'2017-12-31'][data2017.columns[1]]
peaking_power2018 = data2018['2018-01-01':'2018-12-31'][data2018.columns[1]]    
peaking_power = pd.concat([peaking_power2017, peaking_power2018])
peaking_power_group = [peaking_power[peaking_power.index.weekday==i] for i in range(7)]
temperature2017 = (taipei_temperature['2017']['T Min'] + Kaohsiung_temperature['2017']['T Min']) / 2
temperature2018 = (taipei_temperature['2018']['T Min'] + Kaohsiung_temperature['2018']['T Min']) / 2
temperature = pd.concat([temperature2017, temperature2018])
temperature_group = [temperature[temperature.index.weekday==i] for i in range(7)]
group_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

regression = [None] * 7
for i in range(7):
    regression[i] = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=42))
    regression[i].fit(np.array(temperature_group[i]).reshape(-1, 1), peaking_power_group[i])

predict_weekdays = [1, 2, 5, 6, 6, 6, 0]
predict_temperature = (Future_temperature['Taipei T Min'] + Future_temperature['Kaohsiung T Min']) / 2
predict_days = predict_temperature.index
predicted_peaking_Power = pd.Series([0] * 7, index=predict_days)
for i in range(7):
    X = np.array(predict_temperature[predict_days[i]]).reshape(-1, 1)
    predicted_peaking_Power[predict_days[i]] = regression[predict_weekdays[i]].predict(X)
predition = pd.DataFrame({'date': [20190402 + i for i in range(7)], 'peak_load(MW)': np.array(predicted_peaking_Power)})

with open('submission.csv','w') as f:
    predition.to_csv(f, index=False)
    