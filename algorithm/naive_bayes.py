from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from pre_processing import stemmed_data, stemming
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#pandas for making the dataframe

news_data, answer_label = stemmed_data()

vectorizer = TfidfVectorizer()
news_data = vectorizer.fit_transform(news_data)

X_train, X_test, Y_train, Y_test = train_test_split(news_data, answer_label, test_size = 0.2, random_state=3)
nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

    
# accuracy score on the test data
X_test_prediction = nb_model.predict(X_test)
test_data_accuracy_naive_bayes = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_naive_bayes)

def predict_using_naive_bayes(text):
  new_texts = [stemming(text)] # Ensure new_texts is a list of strings
  news_data = vectorizer.transform(new_texts)
  prediction = nb_model.predict(news_data)
  if (prediction[0]=='0'):
    return 'The news is Real 🤩'
  else:
    return 'The news is Fake 👎'