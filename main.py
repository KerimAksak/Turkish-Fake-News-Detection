"""
-*- coding: utf-8 -*-
Spyder Editor
author: KerimAksak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Veri setlerinin okunması
fake_data = pd.read_csv("all_data_lie.csv")
true_data = pd.read_csv("all_data_true.csv")

#%% Veri Setleri Boyutları

print("YALAN HABER SAYISI..:", fake_data.shape)
print("DOĞRU HABER SAYISI..:", true_data.shape)

#%% # Veri Temizleme ve Ön Hazırlık 
#Verilen birleştirilmesi
all_data = pd.concat([fake_data, true_data]).reset_index(drop = True)
print("\nTOPLAM ALIŞAN VERİ SETİ HABER SAYISI..:", all_data.shape)

#%% Verilen karıştılması
from sklearn.utils import shuffle
all_data = shuffle(all_data)
all_data = all_data.reset_index(drop=True)

#%% Birleştirilmiş veri bilgisi
print("Tüm veri bilgisi...:\n", all_data.head())

#%% Veriler içinden tarih, user_id ve user_screen_name sütunlarını kaldırıyoruz. (Analiz için gereksiz)
print("\n\n")
all_data.drop(["date","user_id","user_screen_name"],axis=1,inplace=True)
print("Temizlenmiş veri bilgisi...:\n", all_data.head())

# %% Tweet metinlerinde kelime hatalarını düzeltme.
# Bu işlem için github.com/brolin59/trnlp/ kütüphanesi kullanılmaktadır.

from trnlp import SpellingCorrector
obj = SpellingCorrector()
temp_tweet_data = all_data['tweet']
all_data = all_data.drop(columns="tweet")
dogru_cumle_listesi = [] 
for i in range(len(temp_tweet_data)):
    obj.settext(temp_tweet_data[i])
    dogru_kelimeler = obj.correction(deasciifier=True)
    dogru_cumle = ""
    for j in range(len(dogru_kelimeler)):
        dogru_cumle = dogru_cumle+ " "+dogru_kelimeler[j][0]
        if j == len(dogru_kelimeler)-1:
            dogru_cumle_listesi.append(dogru_cumle)

# pd özelliklerini kullanabilmek için list convert to DataFrame
dogru_cumle_listesi = pd.DataFrame(dogru_cumle_listesi)
# DataFrame columns name ekleme
dogru_cumle_listesi.columns = ["tweet"]

all_data["tweet"] = dogru_cumle_listesi

del i,j,dogru_cumle_listesi,dogru_kelimeler,temp_tweet_data,dogru_cumle,obj

# Ornek
#   all_data["tweet"][1] = " Aşı otizme yol acan bir sey."
#   tweets = all_data['tweet'][1]
#   DÜZELTME İSLEMİ
#   all_data["tweet"][1] = " Aşı otizme yol açan bir şey."


#%% Metin temizleme; tekillik sağlanması için tüm harfler lower_case()
all_data['tweet'] = all_data['tweet'].apply(lambda x: x.lower())
# Ornek
#   all_data["tweet"][1] = " Aşıya HAYIR."
#   tweets = all_data['tweet'][1]
#   DÜZELTME İSLEMİ
#   all_data["tweet"][1] = " Aşıya hayır."

#%% Noktalama işaretlerini temizleme
print("\n")
import string

def isaretlerden_temizle(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

all_data['tweet'] = all_data['tweet'].apply(isaretlerden_temizle)

print("lower_case ve noktalama işaretlerinden temizlenmiş veri..:\n", all_data.head())

#%% # Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('turkish')

all_data['tweet'] = all_data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#%% KelimeBulutu çıktısında konuyla alakasız kelimelerin belirlenmesi ve temizlenmesi.
# bunlar veriyi olumsuz etkileyeceği için kelime temizleme işlemlerinde tekrar
# bir ön eleme gerekmektedir.
def kelime_temizleme(tweet, yasak_kelime):
    import re
    temp = ""
    split_tweet = tweet.split(" ")
    tweet_clear = []
    for i in range(len(split_tweet)):
        for yasak in yasak_kelime:
            if split_tweet[i] == yasak:
                split_tweet[i] = "xxxx"
    
    for i in range(len(split_tweet)):
        patt = re.compile("xxxx")
        temp = patt.sub('', split_tweet[i])
        tweet_clear.append(temp)
    return tweet_clear

def listToString(liste):
    temp = ""
    for text in liste:
        temp = temp +" "+ text
    return temp

tweet = all_data["tweet"]
rt_count = all_data["rt_count"]
like_count = all_data["like_count"]
label = all_data["label"]
speaker = all_data["speaker"]


yasak_kelime = ["rt","http","https","t"]
temizlenmis_tweet_listesi = []

for i in range(len(tweet)):
    yolla = kelime_temizleme(tweet[i],yasak_kelime)
    yolla_string = listToString(yolla)
    temizlenmis_tweet_listesi.append(yolla_string)

all_data = all_data.drop(columns=["tweet","rt_count","like_count","label","speaker"])
temizlenmis_tweet_listesi = pd.DataFrame(temizlenmis_tweet_listesi)
all_data["tweet"] = temizlenmis_tweet_listesi
all_data["rt_count"] = rt_count
all_data["like_count"] = like_count
all_data["label"] = label
all_data["speaker"] = speaker


del i, label, tweet, rt_count, like_count, speaker

#%% Kaç tane yalan ve doğru haber var?
print("\n")
print(all_data.groupby(['label'])['tweet'].count())
all_data.groupby(['label'])['tweet'].count().plot(kind="bar")
plt.show()


#%% Yalan haber kelime bulutu
# conda install: conda install -c conda-forge wordcloud
from wordcloud import WordCloud


fake_data_bilincli = all_data[all_data["label"] == "yalan-bilincli"]
fake_data_bilincsiz = all_data[all_data["label"] == "yalan-bilincsiz"]
fake_data = fake_data_bilincli.append(fake_data_bilincsiz)
all_words_yalan = ' '.join([tweet for tweet in fake_data.tweet])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words_yalan)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del all_words_yalan

#%% Dogru haber kelime bulutu
from wordcloud import WordCloud

real_data = all_data[all_data["label"] == "dogru-bilincli"]
all_words = ' '.join([tweet for tweet in real_data.tweet])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del all_words

#%%
# En fazla tekrara sahip kelimeler (https://www.kaggle.com/rodolfoluna/fake-news-detector)   
import nltk
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

# Most frequent words in fake news
counter(all_data[all_data["label"] == ("yalan-bilincli" or "yalan-bilincsiz") ], "tweet", 20)
#counter(all_data[all_data["label"] == "yalan" ], "tweet", 20)

# Most frequent words in real news
counter(all_data[all_data["label"] == "dogru-bilincli"], "tweet", 20)
#counter(all_data[all_data["label"] == "dogru"], "tweet", 20)

#%% Modelling
# Confusion matrix gösterimi 
# (https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#%% 
all_data = all_data.drop(columns=["rt_count","like_count"])
        
#%%
# Split the data
X_train,X_test,y_train,y_test= train_test_split(all_data["tweet"], all_data.label, test_size=0.3, random_state=42)


#%%
# Vectorizing and applying TF-IDF
print("\n LogisticRegression \n")
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(X_train, y_train)


# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

cm = metrics.confusion_matrix(y_test, prediction, labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])
plt.figure()
plot_confusion_matrix(cm, classes=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])

from sklearn.metrics import classification_report
print(classification_report(y_test, prediction, labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"]))

#%% decision tree

print("\n\n\n\n Decision Tree \n\n")

from sklearn.tree import DecisionTreeClassifier

# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


cm = metrics.confusion_matrix(y_test, prediction,labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])
plt.figure()
plot_confusion_matrix(cm, classes=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])

from sklearn.metrics import classification_report
print(classification_report(y_test, prediction, labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"]))


#%% Random forest
print("\n\n\n\n Random Forest Classifier \n")
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

cm = metrics.confusion_matrix(y_test, prediction,labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])
plt.figure()
plot_confusion_matrix(cm, classes=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"])

from sklearn.metrics import classification_report
print(classification_report(y_test, prediction, labels=["yalan-bilincli", "yalan-bilincsiz", "dogru-bilincli"]))





