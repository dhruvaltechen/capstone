# Importing necessary modules from scikit-learn library:
# LogisticRegression: To create a logistic regression model for classification
# accuracy_score: To evaluate the accuracy of the model's predictions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pre_processing import stemmed_data
#pandas for making the dataframe

news_data, answer_label = stemmed_data()

vectorizer = TfidfVectorizer()
news_data = vectorizer.fit_transform(news_data)

X_train, X_test, Y_train, Y_test = train_test_split(news_data, answer_label, test_size = 0.2, random_state=3)

# Creating an instance of the Logistic Regression model:
# This model will be used to classify the news data into the appropriate categories
model = LogisticRegression()

# Training the model:
# The model is trained using the training data (news_data_train) and the corresponding labels (answer_label_train)
# This will allow the model to learn the relationship between the features and the target labels

model.fit(X_train, Y_train)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy_logistic_regression = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_logistic_regression)

def predict_using_logistic_regression(text):
  new_texts = [text] # Ensure new_texts is a list of strings
  news_data = vectorizer.transform(new_texts)
  prediction = model.predict(news_data)
  print(prediction[0], 'lll')
  if (prediction[0]=='0'):
    return 'The news is Real 🤩'
  else:
    return 'The news is Fake 👎'
