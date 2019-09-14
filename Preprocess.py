import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import zscore

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
            if 'number' in features[i]:
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
def get_data(filePath, labelEncode=False, hotEncode=False, skewness=False):

	diab_df = pd.read_csv('diabetic_data.csv') 
	print('Original data shape {}'.format(diab_df.shape))
	# Create target column
	diab_df.readmitted = diab_df.readmitted.apply(lambda x: 'Yes' if x in ['<30'] else 'No')

	# Missing data
	print('Missing data')
	diab_df.drop('weight',1,inplace=True)
	diab_df.drop('payer_code',1,inplace=True)
	diab_df.medical_specialty.replace('?','Missing',inplace=True)
	diab_df.race.replace('?','Missing',inplace=True)
	diab_df.diag_1.replace('?','Missing',inplace=True)
	diab_df.diag_2.replace('?','Missing',inplace=True)
	diab_df.diag_3.replace('?','Missing',inplace=True)


	# Delete multipule encounters
	print('Multipule encounters')
	temp_df = diab_df.groupby('patient_nbr')['encounter_id'].min().reset_index()
	temp_df = pd.merge(temp_df,diab_df.drop('patient_nbr',1),'left',left_on='encounter_id',right_on='encounter_id')
	temp_df = temp_df[~temp_df['discharge_disposition_id'].isin([11,13,14,19,20,21])]
	temp_df.drop('patient_nbr',1,inplace=True)
	temp_df.drop('encounter_id',1,inplace=True)

	# Transform nominal columns to string type
	print('Transform features')
	temp_df.admission_type_id = temp_df.admission_type_id.astype(str)
	temp_df.discharge_disposition_id = temp_df.discharge_disposition_id.astype(str)
	temp_df.admission_source_id = temp_df.admission_source_id.astype(str)

	# Check outliers
	num_cols = temp_df.dtypes[temp_df.dtypes != "object"].index
	z = np.abs(zscore(temp_df[num_cols]))
	row, col = np.where(z > 4)
	df = pd.DataFrame({"row": row, "col": col})
	rows_count = df.groupby(['row']).count()

	outliers = rows_count[rows_count.col > 2].index
	# There are three rows have more than 2 features that have z-score higher than 4

	# Reduce skewness
	if skewness:
		num_cols = temp_df.dtypes[temp_df.dtypes != "object"].index
		skewed_cols = temp_df[num_cols].apply(lambda x: skew(x))
		skewed_cols = skewed_cols[abs(skewed_cols) > 0.75]
		skewed_features = skewed_cols.index

		for feat in skewed_features:
		    #temp_df[feat] = boxcox1p(temp_df[feat], boxcox_normmax(temp_df[feat]+1))
		    temp_df[feat] = np.log1p(temp_df[feat])
	# Get target column
	Y = temp_df['readmitted']
	temp_df.drop('readmitted',1,inplace=True)

	# Dummify
	if hotEncode:
		cate_col = temp_df.dtypes[temp_df.dtypes == object].index
		dummies_drop = [i + '_'+ temp_df[i].value_counts().index[0] for i in cate_col]
		temp_df = pd.get_dummies(temp_df)
		temp_df.drop(dummies_drop,axis=1,inplace=True)
		print('Data shape after preprocessing: {}'.format(temp_df.shape))
		

	# LabelEncoder
	if labelEncode:
		cate_col = temp_df.dtypes[temp_df.dtypes == object].index
		# process columns, apply LabelEncoder to categorical features
		for i in cate_col:
		    lbl = LabelEncoder() 
		    lbl.fit(list(temp_df[i].values)) 
		    temp_df[i] = lbl.transform(list(temp_df[i].values))
	print('Data shape after preprocessing: {}'.format(temp_df.shape))

	return temp_df,Y







