import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# TRAIN_DATA_PATH = "/train.csv"
# VALID_DATA_PATH = "/validation_data.csv"
# TEST_DATA_PATH = "/comments_to_score.csv"

# df_train2 = pd.read_csv(TRAIN_DATA_PATH)
# df_valid2 = pd.read_csv(VALID_DATA_PATH)
# df_test2 = pd.read_csv(TEST_DATA_PATH)
# cat_mtpl = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
#             'insult': 0.64, 'severe_toxic': 1.5, 'identity_hate': 1.5}
# for category in cat_mtpl:
#     df_train2[category] = df_train2[category] * cat_mtpl[category]

# df_train2['score'] = df_train2.loc[:, 'toxic':'identity_hate'].mean(axis=1)

# df_train2['y'] = df_train2['score']

# min_len = (df_train2['y'] > 0).sum()  # len of toxic comments
# df_y0_undersample = df_train2[df_train2['y'] == 0].sample(n=min_len, random_state=41)  # take non toxic comments
# df_train_new = pd.concat([df_train2[df_train2['y'] > 0], df_y0_undersample])  # make new df
# from tokenizers import (
#     decoders,
#     models,
#     normalizers,
#     pre_tokenizers,
#     processors,
#     trainers,
#     Tokenizer,
# )

# raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# raw_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# raw_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
# from datasets import Dataset

# dataset = Dataset.from_pandas(df_train_new[['comment_text']])

# def get_training_corpus():
#     for i in range(0, len(dataset), 1000):
#         yield dataset[i : i + 1000]["comment_text"]

# raw_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# from transformers import PreTrainedTokenizerFast

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=raw_tokenizer,
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]",
#     mask_token="[MASK]",
# )
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import Ridge

# def dummy_fun(doc):
#     return doc

# labels = df_train_new['y']
# comments = df_train_new['comment_text']
# tokenized_comments = tokenizer(comments.to_list())['input_ids']

# vectorizer = TfidfVectorizer(
#     analyzer = 'word',
#     tokenizer = dummy_fun,
#     preprocessor = dummy_fun,
#     token_pattern = None)

# comments_tr = vectorizer.fit_transform(tokenized_comments)

# regressor = Ridge(random_state=42, alpha=0.8)
# regressor.fit(comments_tr, labels)

# less_toxic_comments = df_valid2['less_toxic']
# more_toxic_comments = df_valid2['more_toxic']

# less_toxic_comments = tokenizer(less_toxic_comments.to_list())['input_ids']
# more_toxic_comments = tokenizer(more_toxic_comments.to_list())['input_ids']

# less_toxic = vectorizer.transform(less_toxic_comments)
# more_toxic = vectorizer.transform(more_toxic_comments)

# # make predictions
# y_pred_less = regressor.predict(less_toxic)
# y_pred_more = regressor.predict(more_toxic)

# print(f'val : {(y_pred_less < y_pred_more).mean()}')
# texts = df_test2['text']
# texts = tokenizer(texts.to_list())['input_ids']
# texts = vectorizer.transform(texts)

# df_test2['prediction'] = regressor.predict(texts)
# df_test2 = df_test2[['comment_id','prediction']]

# df_test2['score'] = df_test2['prediction']
# df_test2 = df_test2[['comment_id','score']]

# df_test2.to_csv('./submission.csv', index=False)
# df2=pd.read_csv('../input/ruddit-jigsaw-dataset/Dataset/Ruddit.csv')
# df2.head()
df2=pd.read_csv('../input/ruddit-jigsaw-dataset/Dataset/ruddit_with_text.csv')
df2.shape
df2.head()

df2['offensiveness_score']

df2['txt'][5]


# dfextreme=df2[df2.offensiveness_score>0].reset_index()

# dfextreme.shape

# samptext1=dfextreme['txt'][0]


def html_remover(data):
    beauti = BeautifulSoup(data,'html.parser')
    return beauti.get_text()

def remove_round_brackets(data):
    return re.sub('\(.*?\)','',data)

def url_remover(data):
    return re.sub(r'https\S','',data)

def remove_punc(data):
    trans = str.maketrans('','', string.punctuation)
    return data.translate(trans)

def white_space(data):
      return ' '.join(data.split())

def remove_non_ascii(data):
    new_words = []
    for word in data:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def text_lower(data):
    return data.lower()

def contraction_replace(data):
    return contractions.fix(data)

def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:
    # if the word is digit, converted to 
    # word else the sequence contineus
        if i.isdigit():
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    temp_str = ' '.join(string)
    return temp_str


def web_associated(data):
    text = html_remover(data)
    text = url_remover(text)
    return text


def complete_noise(data):
    new_data = remove_round_brackets(data)
    new_data = remove_punc(new_data)
    new_data = white_space(new_data)
    return new_data
# corpus2 = [complete_noise(x) for x in corpus1]

import nltk
# import contractions
# import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re, string, unicodedata



def preprocessc(samptext1):
    protext=complete_noise((web_associated(samptext1)))
    return protext


comments=list(df2['txt'])
y=list(df2['offensiveness_score'])


procomments=[preprocessc(com) for com in comments]
len(procomments)
procomments
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
procomments, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vec = TfidfVectorizer(min_df=3, max_df=0.5,analyzer = 'char_wb', ngram_range = (1,5))
X_train = vec.fit_transform(X_train)
X_test=vec.transform(X_test)
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# vec = TfidfVectorizer(min_df= 3, max_df=0.5,analyzer = 'word', ngram_range = (3,5))
# X_train = vec.fit_transform(X_train)
# X_test=vec.transform(X_test)
procomments[0]


X_train[0]


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vec = TfidfVectorizer(min_df=3,max_df=0.5,analyzer = 'char_wb', ngram_range = (2,2))
X = vec.fit_transform(procomments)


dfextreme.head()

# scores=list(dfextreme['offensiveness_score'][0:50])


from sklearn.linear_model import Ridge, Lasso, BayesianRidge
model = Ridge(alpha=0.5)
model.fit(X_train,y_train)


preds = model.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test,preds)

import scipy

scipy.stats.pearsonr(y_test,preds)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,preds)

results=[]
for i in range(1,6):
    for j in range(i,6):
#         from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
        procomments, y, test_size=0.2, random_state=42)
        vec = TfidfVectorizer(min_df=3, max_df=0.5,analyzer = 'char_wb', ngram_range = (i,j))
        X_train = vec.fit_transform(X_train)
        X_test=vec.transform(X_test)
        model = Ridge(alpha=0.5)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        r1=scipy.stats.pearsonr(y_test,preds)
        r2=mean_squared_error(y_test,preds)
        results.append([(i,j),r1[0],r2])


for b in results:
    print(b)

np.array(results)

dfres=pd.DataFrame(data=results,columns=['n-gram range','PCC','MSE'])

dfres.to_csv('reg_results.csv',index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import math


catresults=[]
for i in range(3,6):
    for j in range(i,6):
#         from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
        procomments, y, test_size=0.2, random_state=42)
        vec = TfidfVectorizer(min_df=3, max_df=0.5,analyzer = 'char_wb', ngram_range = (i,j))
        X_train = vec.fit_transform(X_train)
        X_test=vec.transform(X_test)
        catmodel=CatBoostRegressor(loss_function='RMSE')
        catmodel.fit( X_train, y_train,
               eval_set=(X_test, y_test),plot=False)
        catpreds=catmodel.predict(X_test)
        r1=scipy.stats.pearsonr(y_test,catpreds)
        r2=mean_squared_error(y_test,catpreds)
        catresults.append([(i,j),r1[0],r2])
dfrescat=pd.DataFrame(data=catresults,columns=['n-gram range','PCC','MSE'])
dfrescat.to_csv('cat_results.csv',index=False)

catresults

dfrescat=pd.DataFrame(data=catresults,columns=['n-gram range','PCC','MSE'])
dfrescat.to_csv('cat_results.csv',index=False)

# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

tokenizer2 = BertTokenizer.from_pretrained("bert-base-uncased",model_max_len=100)

tokenizer2

tokenizer2('this is cool',return_tensors="tf",padding='model_max_len')

from transformers import BertTokenizer, TFBertModel

import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TFBertModel.from_pretrained("bert-base-uncased")

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer("Hello, my dog is cute", return_tensors="tf")

model.summary()


model2=tf.keras.models.load_model('./bertmodel1')

model2.summary()

model.save('bertmodel1')

!zip -r bertmodelt1.zip bertmodel1

inputs

model

def nmodel(inputs,model,tokenizer):
    outputs1=model(tokenizer(inputs,return_tensors="tf"))
    return (outputs1.last_hidden_state)

def nmodel(inputs,model,tokenizer):
    outputs1=model(tokenizer(inputs,return_tensors="tf"))
    return tf.math.reduce_sum(outputs1.last_hidden_state,axis=1)/outputs1.last_hidden_state.shape[1]

layer1=tf.keras.Dense(128)

nmodel('hi i am papa',model,tokenizer)

inputs = tokenizer("I am papa. this is cool", return_tensors="tf")

outputs = model(inputs)

tf.keras.Input(shape=(12),dtype=tf.int32)

input_ids=tf.keras.Input(shape=(512),dtype=tf.int32)
token_type_ids=tf.keras.Input(shape=(512),dtype=tf.int32)
attention_mask=tf.keras.Input(shape=(512),dtype=tf.int32)

layer1=model({'input_ids':input_ids,
'token_type_ids':token_type_ids,
'attention_mask':attention_mask})


output1=tf.math.reduce_sum(layer1.last_hidden_state,axis=1)/layer1.last_hidden_state.shape[1]
dense1=tf.keras.layers.Dense(1)(output1)
# dense2=tf.keras.layers.Dense(1)(dense1)

output1=tf.math.reduce_sum(layer1.last_hidden_state,axis=1)/layer1.last_hidden_state.shape[1]
dense1=tf.keras.layers.Dense(128)(output1)
dense2=tf.keras.layers.Dense(1)(dense1)

dense2

model3=tf.keras.Model(inputs=[input_ids,token_type_ids,attention_mask],outputs=dense1)

model3.layers[3].trainable=False

model3.layers[3].trainable

model3.layers[3]

model3.summary()

layer1

inputs

inputs

tokenizer

X_train

inputs = tokenizer(X_train, return_tensors="tf",padding='max_length')
test_inputs = tokenizer(X_test, return_tensors="tf",padding='max_length')

ip1=inputs['input_ids']
ip2=inputs['token_type_ids']
ip3=inputs['attention_mask']
ip1t=test_inputs['input_ids']
ip2t=test_inputs['token_type_ids']
ip3t=test_inputs['attention_mask']


# opt=tf.keras.optimizers.Adam(learning_rate=0.001)
# model3.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=opt)


opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model3.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=opt)

y1=tf.stack(y_train)


history1=model3.fit([ip1,ip2,ip3],y1,epochs=10,batch_size=16)

plt.plot(history1.history['loss'])

preds1=model3.predict([ip1t,ip2t,ip3t])

tpreds=[]
for x in preds1:
    tpreds.append(x[0])

tpreds=[]
for x in preds1:
    tpreds.append(x[0])


inputs = tokenizer(["Hello, my dog is cute",'this is rishi'], return_tensors="tf",padding='max')

outputs = model(inputs)

# last_hidden_states = outputs.last_hidden_state

outputs


from transformers import BertModel, BertConfig

# Initializing a BERT bert-base-uncased style configuration

configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration

model = BertModel(configuration)

# Accessing the model configuration

configuration = model.config

configuration







































































