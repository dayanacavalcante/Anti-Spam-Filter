# Imports

import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download()

# Load Data

data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\Anti-Spam Filter\\Data\\spam_or_not_spam.csv', encoding='latin-1')
print(data)

print(data.isnull().sum())

data = data.dropna(how='any', axis=0)
print(data.info())

# EAD

print(data.groupby('label').describe().iloc[0])
print(data.groupby('label').describe().iloc[1])

data['lenght'] = data['email'].apply(len)

data.hist(column='lenght', by='label', bins=20, figsize=(15,6))
plt.show()

# Data Processing

def ProcessText(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    cleanWords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return cleanWords

data['email'].apply(ProcessText)

# Separate training and test data

email_train,email_test,label_train,label_test = train_test_split(data['email'], data['label'], test_size=.3, random_state=1)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=ProcessText)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(email_train,label_train)

pred = pipeline.predict(email_test)

print(classification_report(label_test,pred))
