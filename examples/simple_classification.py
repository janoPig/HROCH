import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from HROCH import PHCRegressor
from xgboost import XGBClassifier


X = np.random.normal(loc=0.0, scale=1.0, size=(10000, 4))
y = (0.5*X[:, 0]**2 > 1.5*X[:, 1]**3)*1.0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_predicted = xgb.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)
print(f'XGBClassifier: mse= {test_mse} r2= {test_r2}')
# XGBClassifier: mse= 0.0068 r2= 0.967908930831287

reg = PHCRegressor(timeLimit=5*60.0)
reg.fit(X_train, y_train)

# predict
y_predicted = reg.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)
# get equation
eq = sp.parse_expr(reg.sexpr)
print(f'PHCRegressor: mse= {test_mse} r2= {test_r2} eq= {str(eq)} ')
# produce similar output to this equations, whitch are correct.
# PHCRegressor: mse= 0.0 r2= 1.0 eq= tanh(pow(0.0, -x1**2 + 2.99404971395121322076*x2**3))
# PHCRegressor: mse= 0.0 r2= 1.0 eq= tanh(x4**2*pow(exp(0.665172489451222514759*x1**2/x2**2 - 2*x2), 167201.609294449637437)**2)
# PHCRegressor: mse= 0.0 r2= 1.0 eq= tanh(exp(54965.34684121816*x1**2/x2**2 - 164896.0405236545*x2))
