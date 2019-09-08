import pandas as pd
from sklearn.preprocessing import LabelEncoder

def group_specialty(x):
    """
    Cardiology: 0
    General pratice: 1
    Internal medicine: 2
    Surgery: 3
    Missing: 4
    Others: 5
    """
    if 'Cardiology' in x:
        return 0
    elif 'GeneralPractice' in x:
        return 1
    elif 'InternalMedicine' in x:    
        return 2
    elif 'Surgery' in x:
        return 3
    elif '?' in x:
        return 4
    else: 
        return 5

def group_race(x):
    """ 
    0: Caucasian, 
    1: AfricanAmerican, 
    2: missing, 
    3: others 
    """
    if x in ['Caucasian']:
        return 0
    elif x in  ['AfricanAmerican']:
        return 1
    elif x in ['?']:
        return 2
    else:
        return 3


def group_admission(x):
    """0: by refer, 
    1: from emergency room, 
    2: other 
    """
    if x in [1, 2, 3]:
        return 0
    elif x in [7]:
        return 1
    else:
        return 2
#
def group_discharge(x):
    """
    0: others, 
    1: discharged to home 
    """
    return int(x==1)

def group_age(x):
        """ 
        age 0-30: 0
        age 30-60: 1
        age 60-100: 2
        """
        if x in ['[0-10)', '[10-20)', '[20-30)']:
            return 0
        elif x in ['[30-40)', '[40-50)', '[50-60)']:
            return 1
        else:
            return 2

def group_readmitted(x):
    """
    <30 days: 1
    Others: 0
    """
    if x in ['<30']:
        return 1     
    else:
        return 0
        
def get_data(filePath, labelEncode=False, hotEncode=False):
    
    data = pd.read_csv(filePath)
    #print(data.columns)
    print('raw data shape {}'.format(data.shape))

    
    misList = data.columns[data.isin(['?']).any()].tolist()
 

    # remove duplicate patient_nbr
    idx = data[['encounter_id', 'patient_nbr']].groupby('patient_nbr').min().reset_index().index
    data = data.iloc[idx].drop(columns=['encounter_id', 'patient_nbr'])
    
    # remove expired discharge_disposition_id entries
    idx = data[data['discharge_disposition_id'].apply(lambda x: x in [11,19,20,21])].index
    data.drop(idx, inplace=True)

    # regroup some variables
    data['age'] = data['age'].apply(group_age)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].apply(group_discharge)
    data['race'] = data['race'].apply(group_race)
    data['admission_source_id'] = data['admission_source_id'].apply(group_admission)
    data['medical_specialty'] = data['medical_specialty'].apply(group_specialty)
    data['readmitted'] = data['readmitted'].apply(group_readmitted)


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


