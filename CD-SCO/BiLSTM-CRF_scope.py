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

from gensim.models import FastText

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Get negations
#==================================================
def get_negation_instances(dataset):
    
    data = dataset
    W, W2, W3, W4 = [], [], [], []
    P, P2, P3, P4 = [], [], [], []
    C, C2, C3, C4 = [], [], [], []
    L, L2, L3, L4 = [], [], [], []
    global length

    for i in range(len(data)):
        w, w2, w3, w4 = [], [], [], []
        p, p2, p3, p4 = [], [], [], []
        c, c2, c3, c4 = [], [], [], []
        l, l2, l3, l4 = [], [], [], []
    
        for i2 in range(len(data[i])):
            try:
                if len(data[i][i2]) == 19:
                    w.append(data[i][i2][3])
                    w2.append(data[i][i2][3])
                    w3.append(data[i][i2][3])
                    w4.append(data[i][i2][3])
                    p.append(data[i][i2][5])
                    p2.append(data[i][i2][5])
                    p3.append(data[i][i2][5])
                    p4.append(data[i][i2][5])
    
                    if data[i][i2][7] != '_':
                        c.append(1)
                    else: 
                        c.append(0)
                    
                    if data[i][i2][10] != '_':
                        c2.append(1)
                    else: 
                        c2.append(0)
                    
                    if data[i][i2][13] != '_':
                        c3.append(1)
                    else: 
                        c3.append(0)
                    
                    if data[i][i2][16] != '_':
                        c4.append(1)
                    else: 
                        c4.append(0)
    
                    if data[i][i2][8] != '_':
                        l.append([1, 0])
                    else:
                        l.append([0, 1])
    
                    if data[i][i2][11] != '_':
                        l2.append([1, 0])
                    else:
                        l2.append([0, 1])
    
                    if data[i][i2][14] != '_':
                        l3.append([1, 0])
                    else:
                        l3.append([0, 1])
                    
                    if data[i][i2][17] != '_':
                        l4.append([1, 0])
                    else:
                        l4.append([0, 1])
                    
                    length+=4
    
                elif len(data[i][i2]) == 16:
                    w.append(data[i][i2][3])
                    w2.append(data[i][i2][3])
                    w3.append(data[i][i2][3])
                    p.append(data[i][i2][5])
                    p2.append(data[i][i2][5])
                    p3.append(data[i][i2][5])
    
                    if data[i][i2][7] != '_':
                        c.append(1)
                    else: 
                        c.append(0)
                    
                    if data[i][i2][10] != '_':
                        c2.append(1)
                    else: 
                        c2.append(0)
                    
                    if data[i][i2][13] != '_':
                        c3.append(1)
                    else: 
                        c3.append(0)
    
                    if data[i][i2][8] != '_':
                        l.append([1, 0])
                    else:
                        l.append([0, 1])
    
                    if data[i][i2][11] != '_':
                        l2.append([1, 0])
                    else:
                        l2.append([0, 1])
    
                    if data[i][i2][14] != '_':
                        l3.append([1, 0])
                    else:
                        l3.append([0, 1])
                        
                    length+=3
    
                elif len(data[i][i2]) == 13:
                    w.append(data[i][i2][3])
                    w2.append(data[i][i2][3])
                    p.append(data[i][i2][5])
                    p2.append(data[i][i2][5])
    
                    if data[i][i2][7] != '_':
                        c.append(1)
                    else: 
                        c.append(0)
                    
                    if data[i][i2][10] != '_':
                        c2.append(1)
                    else: 
                        c2.append(0)
                        
                    if data[i][i2][8] != '_':
                        l.append([1, 0])
                    else:
                        l.append([0, 1])
    
                    if data[i][i2][11] != '_':
                        l2.append([1, 0])
                    else:
                        l2.append([0, 1])
    
                    length+=2
    
                elif len(data[i][i2]) == 10:
                    w.append(data[i][i2][3])
                    p.append(data[i][i2][5])
    
                    if data[i][i2][7] == '_':
                        c.append(0)
                    else: 
                        c.append(1)
                    if data[i][i2][8] == '_':
                        l.append([0, 1])
                    else:
                        l.append([1, 0])
    
                    length+=1
    
            except Exception:
                pass
    
        if len(data[i][i2]) == 19:
            W.append(w)
            W2.append(w2)
            W3.append(w3)
            W4.append(w4)
            P.append(p)
            P2.append(p2)
            P3.append(p3)
            P4.append(p4)
            C.append(c)
            C2.append(c2)
            C3.append(c3)
            C4.append(c4)
            L.append(l)
            L2.append(l2)
            L3.append(l3)
            L4.append(l4)
    
        elif len(data[i][i2]) == 16:
            W.append(w)
            W2.append(w2)
            W3.append(w3)
            P.append(p)
            P2.append(p2)
            P3.append(p3)
            C.append(c)
            C2.append(c2)
            C3.append(c3)
            L.append(l)
            L2.append(l2)
            L3.append(l3)
    
        elif len(data[i][i2]) == 13:
            W.append(w)
            W2.append(w2)
            P.append(p)
            P2.append(p2)
            C.append(c)
            C2.append(c2)
            L.append(l)
            L2.append(l2)
    
        elif len(data[i][i2]) == 10:
            W.append(w)
            P.append(p)
            C.append(c)
            L.append(l)
    
    words = W+W2+W3+W4
    pos = P+P2+P3+P4
    cues = C+C2+C3+C4
    labels = L+L2+L3+L4

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
        print ("F1 score on validation set", valid_pre, valid_rec, valid_f1)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            model.save('bilstm_crf.hdf5')
            print ('saved best model')
        else: print ('No progress')
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
    with codecs.open('scope_bilstm_crf_fT.txt','wb','utf8') as store_pred:
        for s, y_sys, y_hat, seq_len in zip(lex, pred_dev, gold_dev, test_seq_len):
            s = [dic_inv['idxs2w'][w] if w in dic_inv['idxs2w'] else '<UNK>' for w in s]
            s = s[0:seq_len]
            y_sys = y_sys[0:seq_len]
            y_hat = y_hat[0:seq_len]
            assert len(s)==len(y_sys)==len(y_hat)
            for _word,_sys,gold in zip(s,y_sys,y_hat):
                _p = list(_sys).index(_sys.max())
                _g = 0 if list(gold)==[1,0] else 1
                store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
            store_pred.write("\n")

#==================================================
# loading datasets
#==================================================
length = 0
lengths = []

training_data = open('./data/sherlock_train.txt').read()
training_data = training_data.split('\n\n')
training_data = [item.split('\n') for item in training_data]
training_data = [[i.split('\t') for i in item] for item in training_data]

validation_data = open('./data/sherlock_dev.txt').read()
validation_data = validation_data.split('\n\n')
validation_data = [item.split('\n') for item in validation_data]
validation_data = [[i.split('\t') for i in item] for item in validation_data]

test_data = open('./data/sherlock_test.txt').read()
test_data = test_data.split('\n\n')
test_data = [item.split('\n') for item in test_data]
test_data = [[i.split('\t') for i in item] for item in test_data]

train_words, train_pos, train_cues, train_labels = get_negation_instances(training_data)
valid_words, valid_pos, valid_cues, valid_labels = get_negation_instances(validation_data)
test_words, test_pos, test_cues, test_labels = get_negation_instances(test_data)
lengths.append(length)

valid_set_len = len(train_words)+len(valid_words)

test_seq_len = []
for seq in test_words:
    test_seq_len.append(len(seq))

print (len(train_words), "negation instances in the train set")
print (len(valid_words), "negation instances in the validation set")
print (len(test_words), "negation instances in the test set")

words_x = train_words+valid_words+test_words
pos_x = train_pos+valid_pos+test_pos
cues_x = train_cues+valid_cues+test_cues
labels_x = train_labels+valid_labels+test_labels

words_x = pad_documents(words_x)
pos_x = pad_documents(pos_x)
cues_x = pad_cues(cues_x)
labels_x = pad_labels(labels_x)

#==================================================
# Loading fastText embeddings
#==================================================
print ('using fastText embeddings')
nmodel = FastText.load('./emb/doyle.fasttext')
embedding_dim = 100
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

# CD-SCO train set
X_train = xwords[0:len(train_words)] 
X_pos_train = xpos[0:len(train_words)]
X_cues_train = xcues[0:len(train_words)]
Y_train = xlabels[0:len(train_words)]

# CD-SCO validation set
X_valid = xwords[len(train_words):valid_set_len]
X_pos_valid = xpos[len(train_words):valid_set_len]
X_cues_valid = xcues[len(train_words):valid_set_len]
Y_valid = xlabels[len(train_words):valid_set_len]

# CD-SCO test set
X_test = xwords[valid_set_len:len(xwords)]
X_pos_test = xpos[valid_set_len:len(xwords)]
X_cues_test = xcues[valid_set_len:len(xwords)]
Y_test = xlabels[valid_set_len:len(xwords)]

#==================================================
# ---------------------- Parameters section -------------------
#==================================================
# Model Hyperparameters
hidden_dims = 400
embeddings_initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)

# ~ # Training parameters
num_epochs = 50
batch_size = 32
best_f1 = 0.0

moremetrics = MoreMetrics()

#==================================================
# ---------------------- training section -------------------
#==================================================
print("training BiLSTM Model")
inputs_w = Input(shape=(sequence_length,), dtype='int32')
inputs_pos = Input(shape=(sequence_length,), dtype='int32')
inputs_cue = Input(shape=(sequence_length,), dtype='int32')

w_emb = Embedding(vocabulary_size+1, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=True)(inputs_w)
p_emb = Embedding(tag_voc_size+1, embedding_dim, embeddings_initializer=embeddings_initializer, input_length=sequence_length, trainable=True)(inputs_pos)
c_emb = Embedding(2, embedding_dim, input_length=sequence_length, embeddings_initializer=embeddings_initializer, trainable=True)(inputs_cue)

sum_emb = keras.layers.add([w_emb, p_emb, c_emb])

dropout_emb = Dropout(0.5)(sum_emb)

BiLSTM = Bidirectional(LSTM(hidden_dims, recurrent_dropout=0.5, return_sequences=True))(dropout_emb)

outputs = CRF(2, sparse_target=False)(BiLSTM)

model = Model(inputs=[inputs_w, inputs_pos, inputs_cue], outputs=outputs)

model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

model.summary()

model.fit([X_train, X_pos_train, X_cues_train], Y_train,  batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=([X_valid, X_pos_valid, X_cues_valid], Y_valid), callbacks=[moremetrics])

#==================================================
# ---------------------- testing section -------------------
#==================================================
print("Using BiLSTM Model on test set")

custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
model = load_model('bilstm_crf.hdf5', custom_objects)

preds = model.predict([X_test, X_pos_test, X_cues_test])

get_eval(preds,Y_test)

store_prediction(X_test, dic_inv, preds, Y_test)
