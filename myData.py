import pandas as pd
from sklearn.preprocessing import LabelEncoder


def group_age(x):
        
        if x in ['[0-10)', '[10-20)', '[20-30)']:
            return '[0-30)'
        elif x in ['[30-40)', '[40-50)', '[50-60)']:
            return '[30-60)'
        elif x in ['[60-70)', '[70-80)', '[80-90)']:
            return '[60-90)'
        else:
            return '[90-100)'

def get_data(filePath, labelEncode=False, hotEncode=False):
    
    data = pd.read_csv(filePath)
    #print(data.columns)
    print('raw data shape {}'.format(data.shape))

    # group age into 4 classes
    data['age'] = data['age'].apply(group_age)

    misList = data.columns[data.isin(['?']).any()].tolist()

    #print('Missing value percentage:')
    #for x in misList:
    #    print('{0:<20} {1:.2f}%'.format(x, data[x].isin(['?']).sum()/data.shape[0]*100))

    # reduce readmitted class from three classes (>30, <30, No) to two classes (Yes, No) 
    data.readmitted = data.readmitted.apply(lambda x: 'Yes' if x in ['<30'] else 'No')

    # remove duplicate patient_nbr
    idx = data[['encounter_id', 'patient_nbr']].groupby('patient_nbr').min().reset_index().index
    data = data.iloc[idx].drop(columns=['encounter_id', 'patient_nbr'])
    
    # remove expired discharge_disposition_id entries
    idx = data[data['discharge_disposition_id'].apply(lambda x: x in [11,19,20,21])].index
    data.drop(idx, inplace=True)


    # convert some categorical variables to numerical variables
    numerical = ['number_diagnoses', 'number_inpatient', 'number_emergency','number_outpatient', 'num_medications','num_procedures','num_lab_procedures']
    for x in data.columns:
        if data[x].dtypes == 'int64':
            numerical.append(x)

    categorical = list(set(data.columns.tolist()).difference(set(numerical)))
    #print(categorical)

    for x in numerical:
        if data[x].dtypes != 'int64':
            data[x] = data[x].astype(int)
            
    # lable encode categorical variables
    if labelEncode:
        le = LabelEncoder()
        for i in categorical:
            data[i] = le.fit_transform(data[i])
    print('processed data shape: {}'.format(data.shape))
    return data


