import pandas as pd
import re
import string
import csv
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.preprocessing import text, sequence
from sklearn.feature_extraction.text import TfidfVectorizer

#txt = pd.read_csv('train-balanced-sarcasm.csv')
txt = pd.read_json('./Sarcasm_Headlines_Dataset.json', lines = True)

stop_words = set(stopwords.words('english'))
punc = list(string.punctuation)
del punc[14] #/
del punc[1] #"
punc=' '.join(punc)

del txt['article_link']

txt = txt[txt['headline'].notnull()]
#txt=text['comment'].to_string()
tokenized_list = []
tagged_list=[]
tagged_comment=[]
new_text = []
new_doc=[]
for i in txt['headline']:
    #txt=i.to_string()
    tokenized = sent_tokenize(i)
    #print(tokenized)
    for s in tokenized:
        tx=''.join(map(str,s))
        wordsList = word_tokenize(tx)
        wordsList = [w for w in wordsList if not w in stop_words]
        tagged = nltk.pos_tag(wordsList)
        tagged_list.append(tagged)
        #print(tagged)
        for word in tagged:
            new_text.append(word[0] + "/" + word[1])

    tagged_comment.append(tagged_list)
    tagged_list.clear()
    doc = ' '.join(new_text)
    new_doc.append(doc)
    new_text.clear()
    doc=''
#print(text['comment'].head())
print(new_doc[2])
#print(new_doc[3])
#print(new_doc[4])

new_doc_opt=[]

for a in new_doc:
    a = re.sub(r'.\d.', ' ', a)
    a = re.sub(r'.\d', ' ', a)
    a = re.sub(r'\d.', ' ', a)
    a = a.translate(str.maketrans(' ', ' ', punc))
    a = re.sub(r' / ', ' ', a)
    new_doc_opt.append(a.lower())

print(new_doc_opt[2])
posdata = []

#preparing text
for i in new_doc_opt:
    posdata.append(i.split())

#tokenization. maximum length of all sequences = 20 pos
tokenizer = text.Tokenizer(num_words=35000)
tokenizer.fit_on_texts(posdata)
tokenized_train = tokenizer.texts_to_sequences(posdata)
xpos = sequence.pad_sequences(tokenized_train, maxlen = 20)

#create ft-idf vectorizer
vectorizer = TfidfVectorizer()

#created the ft-idf vectors on pos
pos_features=vectorizer.fit_transform(new_doc)
