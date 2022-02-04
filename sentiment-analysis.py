import numpy as np
import glob
from re import sub
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Data: https://archive.ics.uci.edu/ml/datasets/Twitter+Data+set+for+Arabic+Sentiment+Analysis
negative_files = glob.glob('sentiment-data/Negative/*.txt')
positive_files = glob.glob('sentiment-data/Positive/*.txt')

def clean_arabic_text(text):
    text = sub('[^ةجحخهعغفقثصضشسيىبلاآتنمكوؤرإأزدءذئطظ]', ' ', text)
    text = sub(' +', ' ', text)
    text = sub('[آإأ]', 'ا', text)
    text = sub('ة', 'ه', text)

    return text

negative_texts = []
positive_texts = []

# Read positive tweets
for file in positive_files:
    with open(file, 'r', encoding='utf-8') as p_f:
        try:
            text = p_f.read()
            text = clean_arabic_text(text)
            if text == '':
                continue
            positive_texts.append(text)
        except UnicodeDecodeError:
            continue

# Read negative tweets
for file in negative_files:
    with open(file, 'r', encoding='utf-8') as n_f:
        try:
            text = n_f.read()
            text = clean_arabic_text(text)
            if text == '':
                continue
            negative_texts.append(text)
        except UnicodeDecodeError:
            continue

positive_labels = [1] * len(positive_texts)
negative_labels = [0] * len(negative_texts)

all_texts = positive_texts + negative_texts
all_labels = positive_labels + negative_labels

all_texts, all_labels = shuffle(all_texts, all_labels)

# Split the data to train and test
x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.20)

# Count Vectorizer
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

vectorizer.fit(x_train)
x_train = vectorizer.fit_transform(x_train)

model = LinearSVC()
model.fit(x_train, y_train)

x_test = vectorizer.transform(x_test)
predictions = model.predict(x_test)

print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')