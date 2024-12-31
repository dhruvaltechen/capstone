from algorithm.logistic_regression import test_data_accuracy_logistic_regression
from algorithm.decision_tree import test_data_accuracy_decision_tree
from algorithm.knn import test_data_accuracy_knn
from algorithm.naive_bayes import test_data_accuracy_naive_bayes
from algorithm.svm import test_data_accuracy_svm

algorithm_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'SVM']

accuracy_scores = [
    test_data_accuracy_logistic_regression * 100, 
    test_data_accuracy_decision_tree * 100, 
    test_data_accuracy_knn * 100, 
    test_data_accuracy_naive_bayes * 100, 
    test_data_accuracy_svm * 100
]