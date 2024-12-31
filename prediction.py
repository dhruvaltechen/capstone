from algorithm.logistic_regression import predict_using_logistic_regression
from algorithm.decision_tree import predict_using_decision_tree
from algorithm.svm import predict_using_svm
from algorithm.knn import predict_using_knn
from algorithm.naive_bayes import predict_using_naive_bayes

def predict(text, algorithm):
    if algorithm == 'logistic_regression':
        print(algorithm)
        return predict_using_logistic_regression(text)
    elif algorithm == 'decision_tree':
        print(algorithm)
        return predict_using_decision_tree(text)
    elif algorithm == 'support_vector_machine':
        return predict_using_svm(text)
    elif algorithm == 'k_nearest_neighbor':
        return predict_using_knn(text)
    elif algorithm == 'naive_bayes':
        return predict_using_naive_bayes(text)