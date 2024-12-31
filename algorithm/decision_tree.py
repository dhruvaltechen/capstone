from sklearn.tree import DecisionTreeClassifier
from train_test_split2 import X_train, Y_train, X_test, Y_test
from sklearn.metrics import accuracy_score

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)


# accuracy score on the test data
X_test_prediction = dt_model.predict(X_test)
test_data_accuracy_decision_tree = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_decision_tree)
