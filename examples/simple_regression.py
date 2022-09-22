import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from HROCH import PHCRegressor


def add_gausian_noise(a: np.array, std: float):
    std = std*np.sqrt(np.mean(a*a))
    noise = np.random.normal(loc=0.0, scale=std, size=a.shape)
    return a + noise


X = np.random.normal(loc=0.0, scale=1.0, size=(10000, 4))
y = 0.5*X[:, 0]**2 + 1.5*X[:, 1] - 1.0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# add noise to train data
y_train = add_gausian_noise(y_train, 0.1)

reg = PHCRegressor()
reg.fit(X_train, y_train)

# predict
y_predicted = reg.predict(X_test)
test_mse = metrics.mean_squared_error(y_predicted, y_test)
test_r2 = metrics.r2_score(y_predicted, y_test)
# get equation
eq = sp.parse_expr(reg.sexpr)

print(f'mse= {test_mse} r2= {test_r2} eq= {str(eq)} ')
# mse= 4.099819574155027e-06 r2= 0.9999985150809174 eq= 0.501029191555295595653*x1**2 + 1.50102919155529559565*x2 - 1.0
