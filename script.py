import pickle
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open("C:\\Users\\oleg.veselov\\Diploma\\scoring.pkl", "rb") as f:
    model = pickle.load(f)
with open("C:\\Users\\oleg.veselov\\Diploma\\one_hot.pkl", "rb") as f:
    one_hot = pickle.load(f)

test_row = [22, 59000, 'RENT', 123.0, 'PERSONAL', 'D', 35000, 16.02, 0.59, 'Y', 3]
df_columns = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']

test_row = pd.DataFrame([test_row], columns=df_columns)
test_row = test_row.drop('loan_grade', axis=1)

cat_vars = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
oh2 = one_hot.transform(test_row[cat_vars])

test_row = test_row.drop(cat_vars, axis=1)
test_row_onehot = test_row.join(pd.DataFrame(oh2.toarray(), columns=one_hot.get_feature_names()))
test_row_onehot



print(model.predict(test_row_onehot))
