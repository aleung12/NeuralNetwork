from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import datetime



def load_data(filename, training=True):

    data = pd.read_csv(filename)

    flight_code = data['flight_no'].to_numpy()
    week        = data['Week'].to_numpy()
    destination = data['Arrival'].to_numpy()
    std_hour    = data['std_hour'].to_numpy()
    flight_date = data['flight_date'].to_numpy()
    
    if training: 
        is_claim = data['is_claim'].to_numpy().reshape(-1,1)
    else:        
        flight_id = data['flight_id'].to_numpy()
    
    N = data.shape[0]

    year         = np.full(N, np.nan, dtype=int)
    month        = np.full(N, np.nan, dtype=int)
    day_of_month = np.full(N, np.nan, dtype=int)
    day_of_week  = np.full(N, np.nan, dtype=int)
    carrier      = np.full(N, '', dtype='S2')
    
    for i in range(N):

        this_date = map(int, flight_date[i].split('-'))
        year[i], month[i], day_of_month[i] = this_date
        day_of_week[i] = datetime.date(*this_date).weekday()    ## Monday=0, ..., Sunday=6

        carrier[i] = flight_code[i][:2]     ## doing this because there are some flights whose "Airline" entry is NULL


    le, ohe = LabelEncoder(), OneHotEncoder(categories='auto')
    encode = lambda x : ohe.fit_transform(le.fit_transform(x).reshape(-1,1)).toarray()
    
    year         = encode(year)
    month        = encode(month)
    day_of_month = encode(day_of_month)
    day_of_week  = encode(day_of_week)
    std_hour     = encode(std_hour)
    carrier      = encode(carrier)
    destination  = encode(destination)

    nn_input = np.concatenate((year.T, month.T, day_of_month.T, day_of_week.T, std_hour.T, carrier.T, destination.T)).T

    print('input shape: {}'.format(nn_input.shape))

    if training: return nn_input, is_claim
    else:        return nn_input, flight_id
