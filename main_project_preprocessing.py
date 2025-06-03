import pandas as pd
from Pre_processing import *

data = pd.read_csv('player-value-prediction.csv')
####################################################
# fill all string columns by mode
columns = ('club_team', 'national_team', 'club_position', 'national_team_position', 'club_join_date',
           'contract_end_year')
data = Feature_Mode(data, columns)
####################################################
data = data.fillna({
    'tags': 'NULL', 'traits': 'NULL', 'LS': 'NULL', 'ST': 'NULL', 'RS': 'NULL', 'LW': 'NULL', 'LF': 'NULL',
    'CF': 'NULL',
    'RF': 'NULL', 'RW': 'NULL', 'LAM': 'NULL', 'CAM': 'NULL', 'RAM': 'NULL', 'LM': 'NULL', 'LCM': 'NULL', 'CM': 'NULL',
    'RCM': 'NULL', 'RM': 'NULL', 'LWB': 'NULL', 'LDM': 'NULL', 'CDM': 'NULL', 'RDM': 'NULL',
    'RWB': 'NULL', 'LB': 'NULL', 'LCB': 'NULL', 'CB': 'NULL', 'RCB': 'NULL', 'RB': 'NULL'
})

# fill all integer columns by avg
cols_avg = data.mean(axis=0)
data = data.fillna(cols_avg)

####################################################
pos = {
    'RES': 1, 'SUB': 1, 'GK': 1, 'CB': 1, 'LCB': 1, 'RCB': 1, 'RB': 1, 'LB': 1, 'RWB': 1, 'LWB': 1, 'CDM': 1,
    'RDM': 1, 'LDM': 1, 'CM': 1, 'RCM': 1, 'LCM': 1, 'RM': 1, 'LM': 1, 'CAM': 1, 'RAM': 1, 'LAM': 1, 'RW': 1,
    'LW': 1, 'RF': 1, 'LF': 1, 'CF': 1, 'RS': 1, 'LS': 1, 'ST': 1, 'NULL': 0, 'High': 3, 'Medium': 2, 'Low': 1
}

# preprocess to (traits and work_rate and positions)
column_spilt = ['traits', 'work_rate', 'positions']
j = 0
for col in column_spilt:
    j = 0
    if col == 'work_rate':
        separator = '/'
    else:
        separator = ','
    for dat in data[col]:
        arr = dat.split(separator)
        i = 0
        dat = 0
        while i < len(arr):
            arr[i] = arr[i].replace(" ", "")
            if col != 'traits':
                if arr[i] in pos:
                    dat = dat + pos[arr[i]]
            else:
                if arr[i] != 'NULL':
                    dat = dat + 1
            i = i + 1
        data[col][j] = dat
        j = j + 1

# preprocess to tags
p = 0
for j in data['tags']:
    va = j.split(',')
    if va[0] == 'NULL':
        data['tags'][p] = 0
    else:
        data['tags'][p] = len(va)
    p = p + 1
#####################################################
# preprocess to club_position
h = 0
for j in data['overall_rating']:
    if j >= 90:
        data['club_position'][h] = 10
    elif 90 > j >= 85:
        data['club_position'][h] = 9
    elif 85 > j >= 80:
        data['club_position'][h] = 8
    elif 80 > j >= 75:
        data['club_position'][h] = 7
    elif 75 > j >= 70:
        data['club_position'][h] = 6
    elif 70 > j >= 65:
        data['club_position'][h] = 5
    elif 65 > j >= 60:
        data['club_position'][h] = 4
    elif 60 > j >= 55:
        data['club_position'][h] = 3
    elif 55 > j >= 50:
        data['club_position'][h] = 2
    elif j < 50:
        data['club_position'][h] = 1
    h = h + 1

####################################################
# last 26 columns
team_pos = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',
            'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for col in team_pos:
    x = 0
    for dat in data[col]:
        if dat != 'NULL':
            arr = dat.split('+')
            value = int(arr[0]) + int(arr[1])
            data[col][x] = value
        else:
            data[col][x] = 40
        x = x + 1
#####################################################
t = 0
for j in data['body_type']:
    if j == 'Normal':
        data['body_type'][t] = 2
    elif j != 'Stocky':
        data['body_type'][t] = 3
    elif j != 'Lean':
        data['body_type'][t] = 1
    else:
        data['body_type'][t] = 2
    t = t + 1
###################################################
u = 0
for dat in data['contract_end_year']:
    va = dat.split('-')
    if len(va) > 1:
        value = 0
        v = '20' + va[2]
        data['contract_end_year'][u] = int(v)
    else:
        value = data['contract_end_year'][u] = int(va[0])
    u = u + 1

# drop columns that is not important
data = data.drop(columns={'id', 'name', 'full_name', 'birth_date', 'nationality', 'height_cm', 'weight_kgs',
                          'club_jersey_number', 'club_join_date', 'national_team', 'national_rating',
                          'national_team_position', 'national_jersey_number', 'club_team', 'preferred_foot'})

data.to_csv('Final_Fifa_Data_after_preprocessing.csv', index=False)
