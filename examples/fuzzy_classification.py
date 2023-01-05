import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from HROCH import PHCRegressor
from xgboost import XGBRegressor


# ((X0 & X15) | (!X3 & X18)) & (X22 | X25)
X = np.random.uniform(low=0.0, high=1.0, size=(10000, 40))
A = X[:, 0] * X[:, 15]
B = (1.0 - X[:, 3]) * X[:, 18]
C = A + B - A * B  # A or b
D = X[:, 22] + X[:, 25] - X[:, 22] * X[:, 25]
y = C * D

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_predicted = xgb.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)
print(f'XGBClassifier: mse= {test_mse} r2= {test_r2}')

reg = PHCRegressor(time_limit=5.0, problem='fuzzy', stopping_criteria=1e-12, random_state=43)
reg.fit(X_train, y_train)

# predict
y_predicted = reg.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)

print(f'PHCRegressor: mse= {test_mse} r2= {test_r2} eq= {str(reg.sexpr)} ')
