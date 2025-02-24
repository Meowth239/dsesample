from read_data import *
from sklearn.linear_model import Lasso

fred = import_data()
print(fred.shape)
y = fred[['CPIAUCSL']]
X = fred[['CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS']]
train_y = y.iloc[0:700]
test_y = y.iloc[700:]
train_X = X.iloc[0:700]
test_X = X.iloc[700:]

clf = Lasso(alpha = .5, max_iter = 10000)
clf.fit(train_X, train_y)
print(clf.score(test_X, test_y))