from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pre_processing import stemmed_data, stemming
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#pandas for making the dataframe

news_data, answer_label = stemmed_data()

vectorizer = TfidfVectorizer()
news_data = vectorizer.fit_transform(news_data)

X_train, X_test, Y_train, Y_test = train_test_split(news_data, answer_label, test_size = 0.2, random_state=3)
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train, Y_train)


# accuracy score on the test data
Y_test_prediction = knn_model.predict(X_test)
test_data_accuracy_knn = accuracy_score(Y_test_prediction, Y_test)

print('Accuracy score of the test data using knn : ', test_data_accuracy_knn)

def predict_using_knn(text):
  new_texts = [stemming(text)] # Ensure new_texts is a list of strings
  news_data = vectorizer.transform(new_texts)
  prediction = knn_model.predict(news_data)
  if (prediction[0]=='0'):
    return 'The news is Real 🤩'
  else:
    return 'The news is Fake 👎'