import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

data = pd.read_csv("data/insurance.csv")

data["is_obese"] = (data["bmi"] >= 30).astype(int)
data["is_older"] = (data["age"] >= 50).astype(int)
data["smoker_obese"] = ((data["smoker"] == "yes") & (data["bmi"] >= 30)).astype(int)
data["smoker_older"] = ((data["smoker"] == "yes") & (data["age"] >= 50)).astype(int)

y = data.charges
X = data[['age','sex','bmi','children','smoker','region', 'is_obese', 'is_older', 'smoker_obese', 'smoker_older']]

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=0)

numcol=['age', 'bmi', 'is_obese', 'is_older', 'smoker_obese', 'smoker_older']
catcol = ['sex', 'children', 'smoker' ,'region']

numpre = SimpleImputer(strategy='median')

catpre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
# q1 = data.charges.quantile(0.25)
# q3 = data.charges.quantile(0.75)
# iqr = q3 - q1

preprocess = ColumnTransformer(transformers=[
    ('num', numpre, numcol),
    ('cat', catpre, catcol)
])
model = XGBRegressor(n_estimators=500, learning_rate = 0.05)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('model', model)
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_val)

# mae = -(cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error"))
# print(mae.mean())

age = int(input("Enter age: "))
sex = input("Enter sex: ").lower()
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = input("Are you a smoker? (yes/no): ").lower()
region = input("Enter region (East/West/North/South): ").lower()
is_obese = 1 
if bmi < 30:
    is_obese = 0

is_older = 1 
if age < 50:
    is_older = 0

smoker_obese = 1 
if (smoker == "no" or bmi < 30): 
    smoker_obese = 0

smoker_older = 1 
if (smoker == "no" or age < 50):
    smoker_older = 0
    
user_input = pd.DataFrame([[age, sex, bmi, children, smoker, region,is_obese, is_older, smoker_obese, smoker_older]],columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region','is_obese', 'is_older', 'smoker_obese', 'smoker_older'])
prediction = pipeline.predict(user_input)
print(f"Predicted Insurance Charge: ${round(prediction[0], 2)}")