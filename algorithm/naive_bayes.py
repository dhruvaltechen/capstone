from sklearn.naive_bayes import MultinomialNB
from train_test_split2 import X_train, Y_train, X_test, Y_test
from sklearn.metrics import accuracy_score

nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

    
# accuracy score on the test data
X_test_prediction = nb_model.predict(X_test)
test_data_accuracy_naive_bayes = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_naive_bayes)