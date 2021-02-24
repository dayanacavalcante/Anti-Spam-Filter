# Anti-Spam Filter

### _The Problem_

Classify emails as spam or non-spam.

### _How?_

With the Naive Bayes algorithm.

### _Exploratory Data Analysis_

Through the _"groupby"_ function, note that there is a predominance of non-spam emails.

```
email  count                               2500
       unique                              2445
       top       url URL date not supplied URL
       freq                                  10
Name: 0, dtype: object
```

Through the histogram note that there are more characters in spam messages:
![](/Chart/Histogram.png)

### _Data Processing_

A function was created to clear the text, removing punctuation, separating the words in vectors, placing all words in lowercase and eliminating the words that are in stopwords.

```
def ProcessText(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    cleanWords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return cleanWords
```
### _Model Training_

Through the _Pipeline_, each stage of the training model was implemented. 
- _CountVectorizer:_ counts how many times each word appears in spam and non-spam emails;
- _TfidfTransformer:_ adjusts the weight of the words;
- _MultinomialNB:_ applies the training of the Naive Bayes algorithm;

```
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=ProcessText)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
```
### Performance Metrics 

The _Classification Report_ metric was applied, which returned a satisfactory result from the model.

```
precision    recall  f1-score   support

           0       0.89      1.00      0.94       750
           1       1.00      0.39      0.56       150

    accuracy                           0.90       900
   macro avg       0.95      0.69      0.75       900
weighted avg       0.91      0.90      0.88       900
```

The data was taken from Kaggle. Follow the link: https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset
