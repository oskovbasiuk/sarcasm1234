import pandas as pd
import stopwords
import re,string
from keras.preprocessing import text
from keras.utils import pad_sequences

#readind dataset
datastore = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

#preparing a variable with stop words and punctuation
stop = set(stopwords.get_stopwords("en"))
punc = list(string.punctuation)
stop.update(punc)


#removing the stopwords from text and making all letters low
def remove_stopw(text):
    result = []
    for i in text.split():
        if i.strip().lower() not in stop:
            result.append(i.strip())
    return " ".join(result)

#removing numbers
def remove_num(text):
    text = re.sub(r'.\d.', '', text)
    text = re.sub(r'.\d', '', text)
    return re.sub(r'\d.', '', text)

#removing punctuation
def remove_pun(text):
    return re.sub(r'[^\w\s]','', text)

#removing the noise in text
def denoise(text):
    text = remove_stopw(text)
    text = remove_num(text)
    text = remove_pun(text)
    return text

datastore['headline']=datastore['headline'].apply(denoise)

#saving colums in values
headlines = datastore['headline']
labels = datastore['is_sarcastic']

headlinesl = datastore['headline'].tolist()
labelsl = datastore['is_sarcastic'].tolist()

del datastore['article_link']

#array for prepared text
datafv = []

#preparing text
for i in datastore.headline.values:
    datafv.append(i.split())

#tokenization. maximum length of all sequences = 20
tokenizer = text.Tokenizer(num_words=35000)
tokenizer.fit_on_texts(datafv)
tokenized_train = tokenizer.texts_to_sequences(datafv)
x = pad_sequences(tokenized_train, maxlen = 20)
"""
#create ft-idf vectorizer
vectorizer = TfidfVectorizer("english")

#created the ft-idf vectors
features = vectorizer.fit_transform(datastore['headline'])
lstm=vectorizer.fit(datastore['headline'])

ftid_train = vectorizer.transform(datastore['headline']).toarray()
"""
def preper(txt):
 dat=[]
 dat.append(txt.split())
 #print("Splitting text")
 #print(dat)
 # tokenization. maximum length of all sequences = 20
 tokenizer = text.Tokenizer(num_words=35000)
 tokenizer.fit_on_texts(dat)
 #print("Tokenizer text")
 #print(tokenizer)
 tokenized_train = tokenizer.texts_to_sequences(dat)
#print("Tokenizer text")
 #print(tokenized_train)
 x = pad_sequences(tokenized_train, maxlen=20)
 return x

