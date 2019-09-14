import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import skew

def metrics_table(y_test, y_estimate):
    y_head=pd.DataFrame([pd.Series(y_estimate, index=y_test.index).sort_values()]).T
    y_head.columns=['proba']

    y_head.reset_index(inplace=True)
    y_head.columns=['id','proba']

    q=list(map(lambda x: x*y_head.shape[0]//10, list(range(1,11))))

    y_head['quantile']=0
    for i,x in enumerate(q):
        if i==0:
            y_head.loc[:x,'quantile']=i
        else:
            y_head.loc[q[i-1]:x, 'quantile']=i

    min_proba = y_head.groupby('quantile')['proba'].min()
    max_proba = y_head.groupby('quantile')['proba'].max()

    pop = y_head.groupby('quantile')['id'].count()

    y_true=y_test.reset_index()

    y_true.columns=['id', 'readmitted']

    y_head = pd.merge(y_true, y_head, how='left', left_on='id', right_on='id')

    readmitted=y_head.groupby('quantile')['readmitted'].sum()
    

    result=pd.DataFrame(index=range(10))

    result['quantile']=list(range(10))

    result['pop']=pop

    result['readmitted']=readmitted

    result['min_proba']=min_proba
    result['max_proba']=max_proba

    result['readmit_pct']=readmitted/pop.sum()

    result['avg_readmit']=readmitted.sum()/pop.sum()

    result['lift']=result['readmit_pct']/result['avg_readmit']*10

    return result

def group_diabet(x):
    
    diag1=x['diag_1']
    diag2=x['diag_2']
    diag3=x['diag_3']
    if ('250' in diag1) or ('250' in diag2) or ('250' in diag3):
        return '1'
    else:
        return '0'


def map_coef(lr, enc, features, inverse_log):
    
    coef_df = pd.Series(lr.coef_[0], index=range(len(lr.coef_[0]))).sort_values(ascending=False)[:20]

    coef_map = []
    for j in coef_df.index:
        total =0
        val = j
        for i,x in enumerate(enc.categories_):
            if total < val and val <= total + len(x):
                #print(val, i, val-total-1)
                coef_map.append((j,i, val-total-1))
                #print('')
                break
            total += len(x)
            #print(i, x, total)
       

    coef = {}
    for idx,i,j in coef_map:
        val = enc.categories_[i][j]
        if inverse_log:
            if ('number' in features[i]) or ('time' in features[i]):
                inv_log = int(np.exp(float(val)))
                key = ('{}_{}'.format(features[i], inv_log))
            else :
                key = ('{}_{}'.format(features[i], val))
            #print(key)
        else:
            key = ('{}_{}'.format(features[i], val))

        value = coef_df[idx]
        coef[key] = value
    return coef

def group_inpatient(x):
    """
    number inpatient <= 1: 0
    others: 1
    """
    if x <=1: 
        return '0'
    else:
        return '1' 

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
        return '0'
    elif 'GeneralPractice' in x:
        return '1'
    elif 'InternalMedicine' in x:    
        return '2'
    elif 'Surgery' in x:
        return '3'
    elif '?' in x:
        return '4'
    else: 
        return '5'

def group_race(x):
    """ 
    0: Caucasian, 
    1: AfricanAmerican, 
    2: missing, 
    3: others 
    """
    if x in ['Caucasian']:
        return '0'
    elif x in  ['AfricanAmerican']:
        return '1'
    elif x in ['?']:
        return '2'
    else:
        return '3'


def group_admission(x):
    """0: by refer, 
    1: from emergency room, 
    2: other 
    """
    if x in [1, 2, 3]:
        return '0'
    elif x in [7]:
        return '1'
    else:
        return '2'
#
def group_discharge(x):
    """
    0: others, 
    1: discharged to home 
    """
    if x==1:
        return '1'
    else:
        return '0'

def group_age(x):
        """ 
        age 0-30: 0
        age 30-60: 1
        age 60-100: 2
        """
        if x in ['[0-10)', '[10-20)', '[20-30)']:
            return '<30'
        elif x in ['[30-40)', '[40-50)', '[50-60)']:
            return '30-60'
        else:
            return '>60'

def group_readmitted(x):
    """
    <30 days: 1
    Others: 0
    """
    if x in ['<30']:
        return 'Yes'     
    else:
        return 'No'

def group_emergency(x):
    """
    group 0: number_emergency < 6
    group 0: number_emergency >= 6
    """
    if x < 6:
        return '0'
    else:
        return '1'
        
def get_data(filePath, labelEncode=False, groupInpatient=False, skewness=False):
    
    data = pd.read_csv(filePath)
    #print(data.columns)
    print('raw data shape {}'.format(data.shape))

    
    #misList = data.columns[data.isin(['?']).any()].tolist()
 

    # remove duplicate patient_nbr
    patient_df=data[['encounter_id', 'patient_nbr']].groupby('patient_nbr').min().reset_index()
    
    data = pd.merge(patient_df[['encounter_id']], data , how='left', left_on='encounter_id', right_on='encounter_id')

    data = data[~data['discharge_disposition_id'].isin([11,13,14,19,20,21])]
    data.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code'], inplace=True)
    
    # regroup some variables
    #data['age'] = data['age'].apply(group_age)
    #data['diabetic'] = data[['diag_1', 'diag_2', 'diag_3']].apply(group_diabet, axis=1)
    data.drop(columns=['diag_1', 'diag_2', 'diag_3'], inplace=True)
    #data['discharge_disposition_id'] = data['discharge_disposition_id'].apply(group_discharge)
    #data['race'] = data['race'].apply(group_race)
    #data['admission_source_id'] = data['admission_source_id'].apply(group_admission)
    #data['medical_specialty'] = data['medical_specialty'].apply(group_specialty)
    data['readmitted'] = data['readmitted'].apply(group_readmitted)
    #data['number_emergency']  = data['number_emergency'].apply(group_emergency)

    # convert to categorical
    data[['admission_type_id','discharge_disposition_id','admission_source_id']] = \
        data[['admission_type_id','discharge_disposition_id','admission_source_id']].astype(str)
    
    # convert some categorical variables to numerical variables
    #numerical = ['number_emergency','number_diagnoses', 'number_inpatient', 'number_outpatient', 
    #    'num_medications','num_procedures','num_lab_procedures']
    numerical = data.columns[data.dtypes=='int64'].tolist()
    print(numerical)
    
    categorical = data.columns[data.dtypes == 'object'].tolist()
    print(categorical)
    if skewness:
        skewed_cols = data[numerical].apply(lambda x: skew(x))
        skewed_cols = skewed_cols[abs(skewed_cols) > 0.75]
        skewed_features = skewed_cols.index

        for feat in skewed_features:
            data[feat] = np.log1p(data[feat])

     
    if groupInpatient:
        data['number_inpatient'] = data['number_inpatient'].apply(group_inpatient)

    # lable encode categorical variables
    le=[]
    if labelEncode:
        
        for x in categorical:
            le.append(LabelEncoder())
            le[-1].fit(data[x])
            data[x] = le[-1].transform(data[x])
    
    target = data['readmitted']
    data.drop(columns=['readmitted'], inplace=True)
    print('processed data shape: {}'.format(data.shape))
   
    return data, target, le


