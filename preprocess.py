#Project part 1
import nltk as nltk
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from wordcloud import WordCloud
# get data 
with open('sample.txt', 'r', encoding='utf8', errors='ignore') as myfile:
    data = myfile.read()
    
split_sentence = nltk.sent_tokenize(data)
word_T = []
for x in split_sentence:
    word_T.append(nltk.word_tokenize(x))
word_T = [re.sub(r'([^\s\w])+', '', w) for x in word_T for w in x]
no_stopwords = [w for w in word_T if w not in nltk.corpus.stopwords.words('english') and w != '']

stemmed, indexes, ind = [], {}, 0
for x in no_stopwords:
    stemmed.append(nltk.stem.PorterStemmer().stem(x))
    indexes.__setitem__(stemmed[ind], ind)
    ind = ind+1

# Word cloud visualization of common words
dict = {}
for x in stemmed:
    count = stemmed.count(x)
    dict.__setitem__(x,count)
word_cloud = WordCloud(background_color="#101010",width=700,height=700)
word_cloud.generate_from_frequencies(dict)
word_cloud.to_file('word_cloud.png')

# 1-hot representation of words
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(stemmed)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
t = open('1hot.txt', 'w')
for i in range(len(onehot_encoded)):
    t.write(str(onehot_encoded[i]))
