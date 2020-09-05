# -*- coding: utf-8 -*-
# Imports
#==================================================
import json, keras, gensim, codecs
import tensorflow as tf
import numpy as np

import keras.preprocessing.text as kpt
from keras.callbacks import Callback
from keras.layers import Dropout, Input, Dense, Embedding, LSTM, GRU, Bidirectional
from keras.models import Model, load_model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from gensim.models import FastText, Word2Vec

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Get negations
#==================================================
def get_negation_instances(dataset):
    
    global length
    data = dataset
    words = []
    pos = []
    cues = []
    labels = []
    for i in range(len(data)):
        w = []
        p = []
        c = []
        l = []
        for i2 in range(len(data[i])):
            try:
                if len(data[i][i2]) > 6:
                    w.append(data[i][i2][2])
                    p.append(data[i][i2][4])
                    if data[i][i2][5] == '_':
                        c.append(0)
                    else: 
                        c.append(1)
                    if data[i][i2][6] == '_':
                        l.append([0, 1])
                    else:
                        l.append([1, 0])
                    length+=1
            except Exception:
                pass
        if len(data[i][i2]) > 6:
            words.append(w)
            pos.append(p)
            cues.append(c)
            labels.append(l)
    return words, pos, cues, labels


# ---------------------- Evaluation -------------------
class MoreMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        global best_f1
        val_predict = model.predict([X_valid, X_pos_valid, X_cues_valid])
        val_targ = Y_valid
        valid_pre, valid_rec, valid_f1 = get_eval_epoch(val_predict,val_targ)
        print ("Precision/recall/F1 score on validation set", valid_pre, valid_rec, valid_f1)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            if lstm:
                if crf:
                    if pretrained:
                        if fasttext:
                            model.save('fasttest_bilstm_crf.hdf5')
                        else :
                            model.save('word2vec_bilstm_crf.hdf5')
                    else:
                        model.save('random_bilstm_crf.hdf5')
                else:
                    if pretrained:
                        if fasttext:
                            model.save('fasttest_bilstm.hdf5')
                        else :
                            model.save('word2vec_bilstm.hdf5')
                    else:
                        model.save('random_bilstm.hdf5')
            else:
                if crf:
                    if pretrained:
                        if fasttext:
                            model.save('fasttest_bigru_crf.hdf5')
                        else :
                            model.save('word2vec_bigru_crf.hdf5')
                    else:
                        model.save('random_bigru_crf.hdf5')
                else:
                    if pretrained:
                        if fasttext:
                            model.save('fasttest_bigru.hdf5')
                        else :
                            model.save('word2vec_bigru.hdf5')
                    else:
                        model.save('random_bigru.hdf5')
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
    
def pad_cues(sentences, padding_word=0):
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
    
def pad_embeddings(w2v_we,emb_size):
    # add at <UNK> random vector at -2 and at -1 for padding
    w2v_we = np.vstack((w2v_we, 0.2 * np.random.uniform(-1.0, 1.0,[2,emb_size])))
    return w2v_we

def get_index(w, _dict, voc_dim):
    return _dict[w] if w in _dict else voc_dim - 2

# ---------------------- storing labeling results -------------------
def store_prediction(lex, dic_inv, pred_dev, gold_dev):
    print ("Storing labelling results for dev or test set...")
    with codecs.open('scope_best_pred.txt','wb','utf8') as store_pred:
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

words, pos, cues, labels = get_negation_instances(data)
lengths.append(length)

words_x = pad_documents(words)
pos_x = pad_documents(pos)
cues_x = pad_cues(cues)
labels_x = pad_labels(labels)

#==================================================
# Parameters
#==================================================
pretrained = False # True to use pretrained embeddings
fasttext = False # True to use fasttext model, False for word2vec
lstm = True # True for LSTM, False for GRU
crf = True # True for CRF prediction, False for softmax

#==================================================
# Loading fastText embeddings
#==================================================
embedding_dim = 100

if pretrained:
    if fasttext:
        print ('using fastText embeddings')
        nmodel = FastText.load('./emb/....fasttext')
    else:
        print ('using Word2Vec embeddings')
        nmodel = Word2Vec.load('./emb/....word2vec')

    word_emb = nmodel.wv.vectors
    idxs2w_list = nmodel.wv.index2word
    pre_w2idxs = dict([(w,i) for i,w in enumerate(idxs2w_list)])
    pre_idxs2w = dict([(v,k) for k,v in pre_w2idxs.items()])

    pad_emb  = pad_embeddings(word_emb,embedding_dim)
    words_idxs = [np.array([get_index(word, pre_w2idxs, pad_emb.shape[0]) for word in doc],dtype=np.int32) for doc in words_x]  
    x_words  = words_idxs[:lengths[0]]
    vocabulary_size = pad_emb.shape[0]
    
    embedding_matrix = np.zeros((len(nmodel.wv.vocab)+3, embedding_dim))
    for i in range(len(nmodel.wv.vocab)):
        embedding_vector = nmodel[nmodel.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    dic_inv = {'idxs2w':pre_idxs2w}

else:
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
xpos = np.array(x_pos, dtype='int32')
xcues = np.array(cues_x, dtype='int32')
xlabels = np.array(labels_x, dtype='int32')
sequence_length = xwords.shape[1]

Xtrain, X_test, Ytrain, Y_test =    train_test_split(xwords, xlabels, random_state=42, test_size=0.2)
X_pos_train,  X_pos_test, _, _   =    train_test_split(xpos, xlabels, random_state=42, test_size=0.2)
X_cues_train, X_cues_test, _, _  =   train_test_split(xcues, xlabels, random_state=42, test_size=0.2)

X_train, X_valid, Y_train, Y_valid =  train_test_split(Xtrain, Ytrain, random_state=42, test_size=0.2)
X_pos_train,  X_pos_valid, _, _   =   train_test_split(X_pos_train, Ytrain, random_state=42, test_size=0.2)
X_cues_train, X_cues_valid, _, _  =   train_test_split(X_cues_train, Ytrain, random_state=42, test_size=0.2)

#==================================================
# ---------------------- Parameters section -------------------
#==================================================
# Model Hyperparameters
hidden_dims = 400
embeddings_initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)

# Training parameters
num_epochs = 50
batch_size = 32
best_f1 = 0.0

moremetrics = MoreMetrics()

#==================================================
# ---------------------- training section -------------------
#==================================================
print("training BiRNN Model")
inputs_w = Input(shape=(sequence_length,), dtype='int32')
inputs_pos = Input(shape=(sequence_length,), dtype='int32')
inputs_cue = Input(shape=(sequence_length,), dtype='int32')

if pretrained:
    w_emb = Embedding(vocabulary_size+1, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=True)(inputs_w)
else:
    w_emb = Embedding(vocabulary_size+1, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_w)

p_emb = Embedding(tag_voc_size+1, embedding_dim, embeddings_initializer=embeddings_initializer, input_length=sequence_length, trainable=True)(inputs_pos)
c_emb = Embedding(2, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_cue)

sum_emb = keras.layers.add([w_emb, p_emb, c_emb])

dropout_emb = Dropout(0.5)(sum_emb)

if lstm:
    BiRNN = Bidirectional(LSTM(hidden_dims, recurrent_dropout=0.5, return_sequences=True))(dropout_emb) #BiLSTM
else:
    BiRNN = Bidirectional(GRU(hidden_dims, dropout=0.5, return_sequences=True))(dropout_emb) #BiGRU

if crf:
    outputs = CRF(2, sparse_target=False)(BiRNN) # CRF prediction layer
    model = Model(inputs=[inputs_w, inputs_pos, inputs_cue], outputs=outputs)
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
else:
    outputs = Dense(2, activation='softmax')(BiRNN) # softmax output layer
    model = Model(inputs=[inputs_w, inputs_pos, inputs_cue], outputs=outputs)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit([X_train, X_pos_train, X_cues_train], Y_train,  batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=([X_valid, X_pos_valid, X_cues_valid], Y_valid), callbacks=[moremetrics])

#==================================================
# ---------------------- testing section -------------------
#==================================================
print("Using BiRNN Model on test set")
custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}

if lstm:
    if crf:
        if pretrained:
            if fasttext:
                model = load_model('fasttest_bilstm_crf.hdf5', custom_objects)
            else :
                model = load_model('word2vec_bilstm_crf.hdf5', custom_objects)
        else:
            model = load_model('random_bilstm_crf.hdf5', custom_objects)
    else:
        if pretrained:
            if fasttext:
                model = load_model('fasttest_bilstm.hdf5')
            else :
                model = load_model('word2vec_bilstm.hdf5')
        else:
            model = load_model('random_bilstm.hdf5')
else:
    if crf:
        if pretrained:
            if fasttext:
                model = load_model('fasttest_bigru_crf.hdf5', custom_objects)
            else :
                model = load_model('word2vec_bigru_crf.hdf5', custom_objects)
        else:
            model = load_model('random_bigru_crf.hdf5', custom_objects)
    else:
        if pretrained:
            if fasttext:
                model = load_model('fasttest_bigru.hdf5')
            else :
                model = load_model('word2vec_bigru.hdf5')
        else:
            model = load_model('random_bigru.hdf5')

preds = model.predict([X_test, X_pos_test, X_cues_test])

get_eval(preds,Y_test)

store_prediction(X_test, dic_inv, preds, Y_test)
