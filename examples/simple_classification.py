import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from HROCH import PHCRegressor
from xgboost import XGBClassifier


X = np.random.normal(loc=0.0, scale=1.0, size=(10000, 4))
y = (0.5*X[:, 0]**2 >= 1.5*X[:, 1]**3)*1.0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_predicted = xgb.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)
print(f'XGBClassifier: mse= {test_mse} r2= {test_r2}')

reg = PHCRegressor(time_limit=5.0, problem={'add': 1.0, 'sub': 1.0, 'mul': 1.0, 'div':0.1, 'lt':1.0, 'gt':1.0, 'lte':1.0, 'gte':1.0})
reg.fit(X_train, y_train)

# predict
y_predicted = reg.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)

print(f'PHCRegressor: mse= {test_mse} r2= {test_r2} eq= {str(reg.sexpr)} ')

