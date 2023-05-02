# https://youtu.be/bwqBZ4IuX7Q


from sklearn.preprocessing import LabelEncoder                            #Label Encoding
from sklearn.feature_extraction.text import CountVectorizer               #Bag of Words            
from sklearn.preprocessing import OneHotEncoder                           #One Hot Encoding
from sklearn.feature_extraction.text import TfidfVectorizer               #TF-IDF        

docs = ["The coffee smells delicious",
        "I need my coffee",
        "My cat loves coffee"
        ]

data, words = [], []
for doc in docs:
    data.append(doc.split())
    words += doc.split()

print(f"Data: {data}")
# Data: [['The', 'coffee', 'smells', 'delicious'], 
#        ['I', 'need', 'my', 'coffee'], 
#        ['My', 'cat', 'loves', 'coffee']]

print(f"Words: {words}")
# Words: ['The', 'coffee', 'smells', 'delicious', 'I', 'need', 'my', 'coffee', 'My', 'cat
# ', 'loves', 'coffee']


# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(words)

print(f"Classes: {label_encoder.classes_}")
# Classes: ['I' 'My' 'The' 'cat' 'coffee' 'delicious' 'loves' 'my' 'need' 'smells']

print("Transform labels to normalized encoding: ", label_encoder.transform(["my", "cat", "need", "coffee"]))
# Transform labels to normalized encoding:  [7 3 8 4]

print("Transform labels back to original encoding: ", label_encoder.inverse_transform([1, 4, 3]))
# Transform labels back to original encoding:  ['My' 'coffee' 'cat']


# One Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data)
print("Categories: ", onehot_encoder.categories_)
# Categories:  [array(['I', 'My', 'The'], dtype=object), 
#               array(['cat', 'coffee', 'need'], dtype=object), 
#               array(['loves', 'my', 'smells'], dtype=object), 
#               array(['coffee', 'delicious'], dtype=object)]
print("Encoded data: ", onehot_encoder.transform(data).toarray())
# Encoded data:  [[0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 1.]
#  [1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0.]
#  [0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0.]]



# Bag of Words
count_vect = CountVectorizer()
count_vect.fit(docs)

print("Vocabulary: ", count_vect.vocabulary_)
# Vocabulary:  {'the': 7, 'coffee': 1, 'smells': 6, 
#               'delicious': 2, 'need': 5, 'my': 4, 
#               'cat': 0, 'loves': 3}

sent = "I need Coffee, Coffee fuels my day"
print(f"BOW representation for {sent}: ", count_vect.transform([sent]).toarray())
# BOW representation for I need Coffee, Coffee fuels my day:  [[0 2 0 0 1 1 0 0]]

count_vect = CountVectorizer(binary=True)
count_vect.fit(docs)

print(f"BOW representation for {sent}: ", count_vect.transform([sent]).toarray())
# BOW representation for I need Coffee, Coffee fuels my day:  [[0 1 0 0 1 1 0 0]]


# Bag of N-Grams
count_vect = CountVectorizer(ngram_range=(1,2))
count_vect.fit(docs)

print("Vocabulary: ", count_vect.vocabulary_)
# Vocabulary:  {'the': 14, 'coffee': 2, 'smells': 12, 'delicious': 4, 
#               'the coffee': 15, 'coffee smells': 3, 
#               'smells delicious': 13, 'need': 10, 'my': 7, 
#               'need my': 11, 'my coffee': 9, 'cat': 0, 'loves': 5, 
#               'my cat': 8, 'cat loves': 1, 'loves coffee': 6}

print(f"BOW representation for 'my cat is lovely': ", count_vect.transform(["my cat is lovely"]).toarray())
# BOW representation for 'my cat is lovely':  [[1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0]]


count_vect = CountVectorizer(ngram_range=(2,2))
count_vect.fit(docs)

print("Vocabulary: ", count_vect.vocabulary_)
# Vocabulary:  {'the coffee': 7, 'coffee smells': 1, 
#               'smells delicious': 6, 'need my': 5,
#               'my coffee': 4, 'my cat': 3, 'cat loves': 0, 
#               'loves coffee': 2}

print(f"BOW representation for 'my cat is lovely': ", count_vect.transform(["my cat is lovely"]).toarray())
# BOW representation for 'my cat is lovely':  [[0 0 0 1 0 0 0 0]]


# TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(docs)

tfidf_vect.transform(["my cat is lovely", "I need Coffee"]).toarray()

print("IDF for all words in the vocabulary: ", tfidf_vect.idf_)
# IDF for all words in the vocabulary:  [1.69314718 1.         1.69314718 1.69314718 1.28768207 1.69314718
#  1.69314718 1.69314718]









