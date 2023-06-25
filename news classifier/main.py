from gensim import models
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)


train_df = pd.read_csv('train_set.csv', sep='\t')
train_tag=train_df['label']
train_rawdata=[data for idx,data in enumerate(train_df['text'])]
train_data=[data.split(' ') for idx,data in enumerate(train_df.iloc[:,1])]

test_df = pd.read_csv('test_a.csv', sep='\t')

test_rawdata=[data for idx,data in enumerate(test_df['text'])]
test_data=[data.split(' ') for idx,data in enumerate(test_df.iloc[:])]

try:
    word2vec_model=models.Word2Vec.load('word2vec.model')
    print('load complete')
except:
    print('no model')
    word2vec_model=models.Word2Vec(sentences=train_data+test_data,vector_size=100,workers=4)
    word2vec_model.save('word2vec.model')


word_vectorizer=TfidfVectorizer(sublinear_tf=True,
strip_accents='unicode',
analyzer='word',
token_pattern=r'\w{1,}',
stop_words='english',
ngram_range=(1, 1),
max_features=10000,
)
word_vectorizer.fit(train_rawdata+test_rawdata)
train_word_features = word_vectorizer.transform(train_rawdata)
test_word_features = word_vectorizer.transform(test_rawdata)



print(word2vec_model.wv['6088'])

X_train = train_word_features
y_train = train_tag
x_train_, x_valid_, y_train_, y_valid_ = train_test_split(X_train, y_train, test_size=0.1)

classifier=LogisticRegression(C=5)
classifier.fit(x_train_,y_train_)

y_pred = classifier.predict(x_valid_)
y_score=classifier.decision_function(x_valid_)
train_scores = classifier.score(x_train_, y_train_)
print(train_scores, f1_score(y_pred, y_valid_, average='macro'))
for i in range(14):
    fpr,tpr,_=roc_curve(y_valid_,y_score[:,i],pos_label=i)
    plt.plot(fpr, tpr,  lw=1)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(y_valid_, y_pred)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

inverse_vocabulary={value:key for key,value in word_vectorizer.vocabulary_.items()}
# _X_train=[]
# for passage in train_word_features:
#     summary_word_vec=np.zeros(100)
#     num=0
#     passage_softmax=softmax(passage.data)
#     for _,idx in enumerate(passage.indices):
#         try:
#             word=inverse_vocabulary[idx]
#             word_vec=word2vec_model.wv[word]
#             summary_word_vec+=(word_vec*passage.data[_])
#             num+=1
#         except:continue
#     summary_word_vec/=len(passage.indices)
#     _X_train.append(summary_word_vec)
#
# x_train_, x_valid_, y_train_, y_valid_ = train_test_split(_X_train, y_train, test_size=0.1)
# classifier_=LogisticRegression(C=5)
# classifier_.fit(x_train_,y_train_)
#
# y_pred = classifier_.predict(x_valid_)
# train_scores = classifier_.score(x_train_, y_train_)
# print(train_scores, f1_score(y_pred, y_valid_, average='macro'))


print('complete')
# for i in range (5):
#     print(train_data[i])

