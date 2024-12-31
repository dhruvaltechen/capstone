#PART 1 PRE-PROCESSING

# Import necessary libraries

#pandas for making the dataframe
import pandas as pd

#stopwords for preprocessing
from nltk.corpus import stopwords

#for stemming
from nltk.stem import PorterStemmer

#for TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

#natural language toolkit
import nltk

#for regular expression
import re


nltk.download('stopwords')

# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('content/news_data.csv')

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')


# merging the author name and news text
news_dataset['content'] = news_dataset['author']+' '+news_dataset['text']


# lets apply the stemming now !!!

# Stemming:

# Stemming is the process of reducing a word to its Root word

# example: actor, actress, acting --> act

# Importing the PorterStemmer for stemming words
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to stem the content (optimized version)
def stemming(content):
    # Clean the text by removing non-alphabetical characters and converting to lowercase
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    
    # Split the text into words
    words = content.split()
    
    # Filter out stopwords and apply stemming in one pass
    stemmed_words = [port_stem.stem(word) for word in words if word not in stop_words]
    
    # Join the words back into a single string
    return ' '.join(stemmed_words)

def stemmed_data():
    # Apply the optimized stemming function
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    
    # Separate the data and labels
    news_data = news_dataset['content']
    answer_label = news_dataset['label']
    
    return news_data, answer_label