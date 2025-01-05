from algorithm.logistic_regression import predict_using_logistic_regression
from algorithm.decision_tree import predict_using_decision_tree
from algorithm.svm import predict_using_svm
from algorithm.knn import predict_using_knn
from algorithm.naive_bayes import predict_using_naive_bayes

def predict(text, algorithm_name):
    if algorithm_name == 'logistic_regression':
        print(algorithm_name)
        return predict_using_logistic_regression(text)
    elif algorithm_name == 'decision_tree':
        print(algorithm_name)
        return predict_using_decision_tree(text)
    elif algorithm_name == 'support_vector_machine':
        return predict_using_svm(text)
    elif algorithm_name == 'k_nearest_neighbor':
        return predict_using_knn(text)
    elif algorithm_name == 'naive_bayes':
        return predict_using_naive_bayes(text)