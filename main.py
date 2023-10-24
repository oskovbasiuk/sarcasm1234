from preprocesing import *
import pandas as pd
import tensorflow as tf
import gensim
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Bidirectional,GRU
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import chardet
#separating the data on the training and testing set using simple tokeniser
x_train, x_test, y_train, y_test = train_test_split(x, datastore.is_sarcastic , test_size = 0.3 , random_state = 0)
#separating the data on the training and testing set using Tfid features
#X_train, X_test, Y_train, Y_test = train_test_split(features, datastore.is_sarcastic, test_size=0.3, random_state=123)

#the dimnesion to which each of the words in the sentence will be encoded to as part of the training
embedding_dim = 100

#maximum length to be retained of each sentence(headline) for the training
max_length = 20

#trucate the sentence from the back if the sentence length exceeds max_length(32)
trunc_type='post'

#pad the sentence with 0's at the back if the sentence length is less than max_length(32)
padding_type='post'

#oov_token = Out Of Vocubulary token to be used if the word is not part of the vocabulary
oov_tok = "<OOV>"

training_size = 20000
"""
#model summery function for Logistic Regression and Naive Bayes classifier
def mod_sum(model,x_test, y_test):
    pred = model.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    print(cm)
    cm = pd.DataFrame(cm, index=['Not Sarcastic', 'Sarcastic'], columns=['Not Sarcastic', 'Sarcastic'])
    plt.figure(figsize=(10, 10), num='Confusion Matrix', facecolor='grey')
    sns.heatmap(cm, cmap="PuRd", linecolor='black', linewidth=2, annot=True, fmt='',xticklabels=['Not Sarcastic', 'Sarcastic'], yticklabels=['Not Sarcastic', 'Sarcastic'])
    plt.show()
    print('Accuracy=',accuracy_score(Y_test, pred)*100,'%')
"""
"""
#function for Logistic Regression and Naive Bayes classifier models accuracy (for table)
def accuracy_(model):
    pred = model.predict(X_test)
    return accuracy_score(Y_test,pred) * 100
"""
#function for Neural Networks models accuracy (for table)
def accuracy(model):
    pred = model.predict(x_test)
    return accuracy_score(y_test, np.where(pred >= 0.5, 1, 0)) * 100

#function for Neural Networks models accuracy (for summary)
def accuracy_lstm(model):
    print("Accuracy on training data is - ", model.evaluate(x_train, y_train)[1] * 100)
    print("Accuracy on testing data is - ", model.evaluate(x_test, y_test)[1] * 100)
    pred = model.predict(x_test)
    return accuracy_score(y_test, np.where(pred >= 0.5, 1, 0))*100
"""
#function for Confusion Matrix for Neural Networks  models
def cm_summary_lstm(model):
    pred = model.predict(x_test)
    cm = confusion_matrix(y_test, np.where(pred >= 0.5, 1, 0))
    print(cm)
    cm = pd.DataFrame(cm, index=['Not Sarcastic', 'Sarcastic'], columns=['Not Sarcastic', 'Sarcastic'])
    plt.figure(figsize=(10, 10), num='Confusion Matrix', facecolor='grey')
    sns.heatmap(cm, cmap="PuRd", linecolor='black', linewidth=2, annot=True, fmt='', xticklabels=['Not Sarcastic', 'Sarcastic'], yticklabels=['Not Sarcastic', 'Sarcastic'])
    plt.show()
"""
"""
#function for epoch plots (Neural Networks models)
def epoch_plot(model):
    history = model.fit(x_train, y_train, batch_size=128, epochs=9, validation_data=(x_test, y_test))
    epoch = [i for i in range(9)]
    fig, ax = plt.subplots(1, 2)
    train_l = history.history['loss']
    train_a = history.history['accuracy']
    val_l = history.history['val_loss']
    val_a = history.history['val_accuracy']
    fig.set_size_inches(20, 10)

    ax[0].plot(epoch, train_l, 'bo-', label='Loss on training data')
    ax[0].plot(epoch, val_l, 'yo-', label='Loss on testing data')
    ax[0].set_title('Training VS Testing Loss')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epoch, train_a, 'bo-', label='Accuracy on training data')
    ax[1].plot(epoch, val_a, 'yo-', label='Accuracy on testing data')
    ax[1].set_title('Training VS Testing Accuracy')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()
"""
#fitting and results for Neural Networks models
def results(model):
    history = model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test), callbacks=[callback])
    #cm_summary_lstm(model)
    accuracy_lstm(model)
    #epoch_plot(model)
"""
#model with Logistic Regression. norm used in the penalization=l1.
#Algorithm to use in the optimization=liblinear (cos this one performs with l1 well)
modellr = LogisticRegression(solver='liblinear', penalty='l1')

#fitting LRmodel with default sets
print(modellr.fit(x_train, y_train))
mod_sum(modellr, x_test, y_test)

#fitting lRmodel with Tfid features
print(modellr.fit(X_train, Y_train))
mod_sum(modellr, X_test, Y_test)

#model with Naive Bayes classifier
#additive smoothing parameter =0.4
modelnb = MultinomialNB(alpha=0.4)

#fitting NBmodel with default sets
print(modelnb.fit(x_train, y_train))
mod_sum(modelnb, x_test, y_test)

#fitting NBmodel with Tfid features
print(modelnb.fit(X_train, Y_train))
mod_sum(modelnb, X_test, Y_test)
"""
#vocabulary size for Neural Networks models
word_size = len(tokenizer.word_index) + 1
word_index=tokenizer.word_index

#defining Early stopping to stop training if validation accuracy does not improve within five epochs
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    verbose=1,
    restore_best_weights=True,
)
"""
#defaul model with lstm and gru
#embeddidng layer using no weights. Length of constant input sequences=20(max_length)
#Long Short Term Memory.dimensionality of the output space=128,
#Fraction of the units to drop for the linear transformation of the recurrent state = 0.3
#Fraction of the units to drop for the linear transformation of the inputs= 0.3
#return the last output
#layer Dense with Activation functions 'sigmoid' and 'relu'
model = Sequential([
    Embedding(word_size, embedding_dim, input_length=max_length, trainable=True),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout = 0.2 , dropout = 0.2)),
    Bidirectional(LSTM(64, return_sequences = True, recurrent_dropout = 0.2 , dropout = 0.2)),
    Bidirectional(GRU(128 , recurrent_dropout = 0.1 , dropout = 0.1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
results(model)
"""
#vocabulary size for weights
vocab_size = len(word_index)

embeddings_index = {}

#store the values into a dictionary. the first value is the word and the second value is the embedding
with open('D:/glove.6B.100d.txt',encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word]=coefs

#initialize a matrix of zeros and then assign the encoding for the words in the vocabulary to the appropriate index in the embedding_matrix
embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for key in sorted(word_index, key=word_index.get)[:vocab_size]:
    embedding_vector = embeddings_index.get(key)
    if embedding_vector is not None:
        embeddings_matrix[word_index[key]] = embedding_vector
embeddings_matrix.shape

#vocab_size = len(word_index)
#helps avoid clutter from old models and layers
tf.keras.backend.clear_session()

#sets the global seed
tf.random.set_seed(51)

#sets the random seed
np.random.seed(51)

#model with lstm and gru
#embeddidng layer using the weights as the embedding matrix that we built earlier. Length of constant input sequences=20(max_length)
#Long Short Term Memory.dimensionality of the output space=128,
#Fraction of the units to drop for the linear transformation of the recurrent state = 0.3
#Fraction of the units to drop for the linear transformation of the inputs= 0.3
#return the last output
#layer Dense with Activation functions 'sigmoid' and 'relu'
modelg = Sequential([
    Embedding(word_size, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=True),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)),
    Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)),
    Bidirectional(GRU(units=128 , recurrent_dropout = 0.1 , dropout = 0.1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelg.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01),loss='binary_crossentropy',metrics=['accuracy'])
results(modelg)

#cleaning weights
del embeddings_matrix

#dimension of future vectors
#size_vec = 200
"""
#creating Word Vectors by Word2Vec (maximum distance between the current and predicted word within a sentence=5
#ignores all words with total frequency lower than 1)
model_V2W = gensim.models.Word2Vec(sentences = datafv , size=embedding_dim , window = 5 , min_count = 1)

#adding 1 cause embedding layer creates one more vector filled with zeros.
word_size = len(tokenizer.word_index) + 1

#creating the weight matrix from word2vec model
def get_weight_matrix(model, word):
    #total vocabulary size plus 1 for new vector
    wordsize = len(word) + 1
    #define weight matrix(fill 0)
    weight_matrix = np.zeros((wordsize, embedding_dim))
    #store vectors pre word
    for word, i in word.items():
        weight_matrix[i] = model[word]
    return weight_matrix

#getting embedding vectors from word2vec model
emb_vectors = get_weight_matrix(model_V2W, tokenizer.word_index)

#helps avoid clutter from old models and layers
tf.keras.backend.clear_session()

#sets the global seed
tf.random.set_seed(51)

#sets the random seed
np.random.seed(51)

#embeddidng layer usings embedding vectors as weights. Length of constant input sequences=20
#Long Short Term Memory.dimensionality of the output space=128,
#Fraction of the units to drop for the linear transformation of the recurrent state = 0.3
#Fraction of the units to drop for the linear transformation of the inputs= 0.3
#return the last output
#layer Dense with Activation function 'sigmoid' and relu
modelvw = Sequential([
    Embedding(word_size, embedding_dim, input_length=max_length, weights=[emb_vectors], trainable=True),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)),
    Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)),
    Bidirectional(GRU(units=128, recurrent_dropout=0.1, dropout=0.1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelvw.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])

#cleaning weigths
del emb_vectors

results(modelvw)
"""
#results
acc=[]
acc=[accuracy(modelg)]
df = pd.DataFrame(
        {
          "Name": ["GloVe"],
           "Accurecy": acc,
        })
print(df)

#mother comes pretty close to using word 'streaming' correctly

#eat your veggies: 9 deliciously different recipes
#w1=[]
#w1='mother comes pretty close to using word streaming correctly'
#w1[1]='eat your veggies: 9 deliciously different recipes'
def calculate(w2):
    if w2=='':
        return 'Input ur text'
    if len(w2)<300:
        w1 = denoise(w2)
        w1 = preper(w1)
            # denoise(w0)
        p1 = modelg.predict(w1)
        return p1 * 100

    return 'text length exceeds 300 symbols'


