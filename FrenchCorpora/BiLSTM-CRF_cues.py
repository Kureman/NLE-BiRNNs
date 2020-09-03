# -*- coding: utf-8 -*-
# Imports
#==================================================
import json, keras, gensim, codecs
import tensorflow as tf
import numpy as np

import keras.preprocessing.text as kpt
from keras.callbacks import Callback
from keras.layers import Dropout, Input, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model, load_model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Get negations
#==================================================
def get_negation_instances(dataset):
    
    global length
    data = dataset
    words = []
    lemmas = []
    pos = []
    labels = []
    for i in range(len(data)):
        w = []
        l = []
        p = []
        c = []
        for i2 in range(len(data[i])):
            try:
                w.append(data[i][i2][2])
                l.append(data[i][i2][3])
                p.append(data[i][i2][4])
                if data[i][i2][5] == '_':
                    c.append([0, 1])
                elif data[i][i2][5] == '***':
                    c.append([0, 1])
                else: 
                    c.append([1, 0])
                length+=1
            except Exception:
                pass
        words.append(w)
        lemmas.append(l)
        pos.append(p)
        labels.append(c)
    return words, lemmas, pos, labels
    
# ---------------------- more metrics -------------------
class MoreMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        global best_f1
        val_predict = model.predict([X_valid, X_lemmas_valid, X_pos_valid])
        val_targ = Y_valid
        valid_pre, valid_rec, valid_f1 = get_eval_epoch(val_predict,val_targ)
        print ("F1 score on validation set", valid_pre, valid_rec, valid_f1)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            model.save('cue_bilstm-crf.hdf5')
            print ('saved best model')
        else: 
            print ('No progress')
        return

def get_eval(predictions,gs):
    y,y_ = [],[]
    for p in predictions: y.extend(map(lambda x: list(x).index(x.max()),p))
    for g in gs: y_.extend(map(lambda x: 0 if list(x)==[1,0] else 1,g))
    print (classification_report(y_,y, digits=4))

def get_eval_epoch(predictions,gs):
    y,y_ = [],[]
    for p in predictions: 
        y.extend(map(lambda x: list(x).index(x.max()),p))

    for g in gs: 
        y_.extend(map(lambda x: 0 if list(x)==[1,0] else 1,g))

    p, r, f1, s = precision_recall_fscore_support(y_,y)
    p_pos = p[0]
    r_pos = r[0]
    f1_pos = f1[0]
    return p_pos, r_pos, f1_pos

# ---------------------- Padding features -------------------
def pad_documents(sentences, padding_word='<PAD>'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pad_labels(sentences, padding_word=[0,1]):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

# ---------------------- storing labeling results -------------------
def store_prediction(lex, dic_inv, pred_dev, gold_dev):
    print ("Storing labelling results for dev or test set...")
    with codecs.open('cue_best_pred.txt','wb','utf8') as store_pred:
        for s, y_sys, y_hat in zip(lex, pred_dev, gold_dev):
            s = [dic_inv.get(word) for word in s]
            assert len(s)==len(y_sys)==len(y_hat)
            for _word,_sys,gold in zip(s,y_sys,y_hat):
                _p = list(_sys).index(_sys.max())
                _g = 0 if list(gold)==[1,0] else 1
                if _word != "<PAD>":
                    store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
            store_pred.write("\n")

#==================================================
# loading datasets
#==================================================
length = 0
lengths = []
data = open('./data/....txt').read()
data = data.split('\n\n')
data = [item.split('\n') for item in data]
data = [[i.split('\t') for i in item] for item in data]

words, lemmas, pos, labels = get_negation_instances(data)
lengths.append(length)

words_x = pad_documents(words)
lemmas_x = pad_documents(lemmas)
pos_x = pad_documents(pos)
labels_x = pad_labels(labels)

#Preparing words without pre-trained embeddings
# ==================================================
# create a new Tokenizer
tokenizer = kpt.Tokenizer(lower=False)

# feed our texts to the Tokenizer
tokenizer.fit_on_texts(words_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index

x_words = [[dictionary[word] for word in text] for text in words_x]

vocabulary_size = len(dictionary)

dic_inv = dict(map(reversed, tokenizer.word_index.items()))

#Preparing lemma without pre-trained embeddings
# ==================================================
# create a new Tokenizer
Lemmatokenizer = kpt.Tokenizer(lower=False)

# feed our texts to the Tokenizer
Lemmatokenizer.fit_on_texts(lemmas_x)

# Tokenizers come with a convenient list of words and IDs
Lemmadictionary = Lemmatokenizer.word_index

x_lemmas = [[Lemmadictionary[word] for word in text] for text in lemmas_x]

lemma_vocabulary_size = len(Lemmadictionary)

#==================================================
# Preparing POS embeddings
# ==================================================
# create a new Tokenizer
postokenizer = kpt.Tokenizer(lower=False)

# feed our texts to the Tokenizer
postokenizer.fit_on_texts(pos_x)

# Tokenizers come with a convenient list of words and IDs
posdictionary = postokenizer.word_index

x_pos = [[posdictionary[pos] for pos in text] for text in pos_x]

tag_voc_size = len(posdictionary)

#==================================================
# Splitting data into the original train, validation and test sets
#==================================================
xwords = np.array(x_words, dtype='int32')
xlemmas = np.array(x_lemmas, dtype='int32')
xpos = np.array(x_pos, dtype='int32')
xlabels = np.array(labels_x, dtype='int32')
sequence_length = xwords.shape[1]

Xtrain, X_test, Ytrain, Y_test =    train_test_split(xwords, xlabels, random_state=42, test_size=0.2)
X_lemmas_train,  X_lemmas_test, _, _   =    train_test_split(xlemmas, xlabels, random_state=42, test_size=0.2)
X_pos_train,  X_pos_test, _, _   =    train_test_split(xpos, xlabels, random_state=42, test_size=0.2)

X_train, X_valid, Y_train, Y_valid =  train_test_split(Xtrain, Ytrain, random_state=42, test_size=0.2)
X_lemmas_train,  X_lemmas_valid, _, _   =   train_test_split(X_lemmas_train, Ytrain, random_state=42, test_size=0.2)
X_pos_train,  X_pos_valid, _, _   =   train_test_split(X_pos_train, Ytrain, random_state=42, test_size=0.2)

# ---------------------- Parameters section -------------------
# Model Hyperparameters
embedding_dim = 100
hidden_dims = 400

# ~ # Training parameters
num_epochs = 20
batch_size = 32
best_f1 = 0.0
embeddings_initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)

moremetrics = MoreMetrics()

#==================================================
# ---------------------- training section -------------------
#==================================================
print("Creating BiLSTM Model")
inputs_w = Input(shape=(sequence_length,), dtype='int32')
inputs_l = Input(shape=(sequence_length,), dtype='int32')
inputs_pos = Input(shape=(sequence_length,), dtype='int32')

w_emb = Embedding(vocabulary_size+1, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_w)
l_emb = Embedding(lemma_vocabulary_size+1, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_l)
p_emb = Embedding(tag_voc_size+1, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_pos)

summed = keras.layers.add([w_emb, l_emb, p_emb])
dropout_emb = Dropout(0.5)(summed)

BiLSTM = Bidirectional(LSTM(hidden_dims, recurrent_dropout=0.5, return_sequences=True))(dropout_emb)

outputs = CRF(2, sparse_target=False)(BiLSTM)

model = Model(inputs=[inputs_w, inputs_l, inputs_pos], outputs=outputs)

model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

model.summary()

model.fit([X_train, X_lemmas_train, X_pos_train], Y_train,  batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=([X_valid, X_lemmas_valid, X_pos_valid], Y_valid), callbacks=[moremetrics])

#==================================================
# ---------------------- testing section -------------------
#==================================================
custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
model = load_model('cue_bilstm-crf.hdf5', custom_objects)

preds = model.predict([X_test, X_lemmas_test, X_pos_test])

get_eval(preds, Y_test)

store_prediction(X_test, dic_inv, preds, Y_test)
