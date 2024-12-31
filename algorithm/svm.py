from sklearn.svm import SVC
from train_test_split2 import X_train, Y_train, X_test, Y_test
from sklearn.metrics import accuracy_score

svm_model = SVC(kernel='linear')  # or 'rbf' kernel
svm_model.fit(X_train, Y_train)


# accuracy score on the test data
X_test_prediction = svm_model.predict(X_test)
test_data_accuracy_logistic_regression = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_logistic_regression)