#!/usr/bin/env python
# coding: utf-8

# # 2. 자연어 처리 개발 준비 

# In[4]:


# 라이브러리 불러오기
import tensorflow as tf


# In[5]:


# 상수값 설정
INPUT_SIZE = (20, 1)
CONV_INPUT_SIZE = (1, 28, 28)
IS_TRAINING = True 

INPUT_SIZE = (20, 1)

inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
ouput = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)




INPUT_SIZE = (20, 1)

inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)
output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid)(hidden)



INPUT_SIZE = (20, 1)

inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
output = tf.keras.layers.Dropout(rate = 0.5)(inputs)



inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate = 0.5)(inputs)
hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(dropout)
output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid)(hidden)



INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.Input(shape = INPUT_SIZE)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu)(input)



INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.Input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu)(inputs)


INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.Input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(inputs)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu)(inputs)
max_pool = tf.keras.layers.MaxPool1D(pool_size = 3, padding = 'same')(conv)
flatten = tf.keras.layers.Flatten()(max_pool)
hidden = tf.keras.layers.Dense(units = 50, activation = tf.nn.relu)(flatten)
output = tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)(hidden)




pip install scikit-learn


# In[3]:


import sklearn
sklearn.__version__


# In[4]:


from sklearn.datasets import load_iris


# In[6]:


iris_dataset = load_iris()
print("iris_dataset key : {}".format(iris_dataset.keys()))


# 키 값을 하나씩 뽑아서 확인

# In[8]:


print(iris_dataset['data'])
print("shape of data : {}".format(iris_dataset['data'].shape))


# 각 데이터마다 4개의 특징(feature) 값을 가지고 있다.   
# 4개의 특징값이 의미하는 바를 확인하기 위해 'feature_names'값을 확인하자 

# In[9]:


print(iris_dataset['feature_names'])


# 이번에는 'target'에 대해 알아보자. 

# In[11]:


print(iris_dataset['target'])
print(iris_dataset['target_names'])


# 이를 통해 알 수 있는 것은  
# - setosa = 0
# - versicolor = 1 
# - virginica = 2    
# 
# 을 라벨값으로 가진다.  
# 
# 마지막으로 'DESCR'에 대해 알아보자. 

# In[13]:


print(iris_dataset['DESCR']) # Description 의 약자, 데이터에 대한 전체적인 요약 정보 제공 


# ### 사이킷런을 이용한 데이터 분리   
# 
# 사이킷런을 이용하면 학습 데이터를 대상으로 학습 데이터와 평가데이터로 쉽게 나눌 수 있다. 

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


train_input, test_input, train_label, test_label = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size= 0.25, random_state = 42)
# random state로 랜덤값 제어 

각 변수의 상태 확인
# In[19]:


print("shape of train_input: {}".format(train_input.shape))
print("shape of test_input: {}".format(test_input.shape))
print("shape of train_label: {}".format(train_label.shape))
print("shape of test_label: {}".format(test_label.shape))


# ### 사이킷런을 이용한 지도 학습   
# - 지도 학습 : 각 데이텅에 대해 정답이 있는 경우 각 데이터의 정답을 예측할 수 있게 학습시키는 과정, 모델이 예측하는 결과를 각 데이터의 정답과 비교해서 모델을 반복적으로 학습    
# 
# 
# **k-최근접 이웃 분류기(K-nearest neigbor classifier)**  
# - 예측하고자 하는 데이터에 대해 가장 가까운 거리에 있는 데이터의 라벨과 같다고 예측하는 방법, 데이터에 대한 사전 지식이 없는 경우 주로 사용  
# - k 값은 예측하고자 하는 데이터와 가까운 몇 개의 데이터를 참고할 것인지를 의미  
# ![KakaoTalk_20201218_203403524.jpg](attachment:KakaoTalk_20201218_203403524.jpg)  
# - k = 1 인 경우에는 가장 가까운 데이터의 라벨값이 Class1이기 때문에 Class1로 예측
# - k = 3 인 경우에는 가까운 3개의 데이터가 Class1 1개, Class2 2개로 구성돼 있기 때문에 이 경우에는 Class2로 예측하게 된다.    
# 
# **k-최근접 이웃 분류기의 특징**  
# + 데이터에 대한 가정이 없어 단순하다. 
# + 다목적 분류와 회귀에 좋다. 
# + 높은 메모리를 요구
# + k값이 커지면 계산이 늦어질 수 있다. 
# + 관련 없는 기능의 데이터의 규모에 민감하다.  
# 
# 이제 k-최근접 이웃 분류기를 직접 만들어보자.

# In[59]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)  # n_neighbors = 1 은 k = 1 이라는 뜻 


# 이제 이렇게 생성한 분류기를 학습 데이터에 적용

# In[60]:


knn.fit(train_input, train_label)


# In[61]:


KNeighborsClassifier(algorithm = 'auto', leaf_size=30, metric='minkowski',
                   metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                   weights='uniform')


# 이제 학습시킨 모델을 사용해 새로운 데이터의 라벨을 예측해보자.  
# 우선은 새롭게 4개의 피처값을 임의로 설정해서 넘파이 배열로 만들자 

# In[62]:


import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])


# 생성한 배열을 보면 꽃받침 길이와 너비가 각각 6.1, 2.8이고 꽃잎의 길이와 너비가 각각 4.7, 1.2인 데이터로 구성돼 있다.  
# 이제 이 값을 대상으로 앞에서 만든 분류기 모델의 predict 함수를 사용해 결과를 예측해보자. 

# In[63]:


knn.predict(new_input)


# 1(versicolor)로 예측하고 있다. 이 데이터는 임의로 만든 것이기 때문에 이 결과가 제대로 예측한 것인지 확인할 수 없다.  
# 이제 모델의 성능을 측성하기 위해 이전에 따로 분리해둔 평가 데이터를 사용해 모델의 성능을 측정해보자. 

# In[64]:


predict_label = knn.predict(test_input)
print(predict_label)


# 이제 예측한 결괏값과 실체 결괏값을 비교해서 정확도가 어느 정도인지 측정해보자. 실제 결과와 동일한 것의 개수를 평균을 구하면 된다. 

# In[65]:


print('test accuracy {:.2f}'.format(np.mean(predict_label == test_label)))


# 정확도가 1.00인데 이는 전체 100%의 정확도로서 매우 좋은 성능을 보여준다. 이것은 데이터 자체가 특징이 라벨에 따라 구분이 잘 되고 모델이 데이터에 매우 적합하다는 것을 의미. 

# ### 사이킷런을 이용한 비지도 학습  
# - 비지도 학습이란 지도학습과는 달리 데이터에 대한 정답, 즉 라벨을 사용하지 않고 만들 수 있는 모델
# - 데이터에 대한 정답이 없는 경우에 적용하기 적합
# 
# **k-평균 군집화(K-means Clustering)모델**을 사용  
# - 군집화란 데이터를 특성에 따라 여러 집단으로 나누는 방법
# - 붓꽃 데이터는 3개의 정답이 있으므로 3개의 군집으로 나누는 방법을 사용
# - k-평균 군집화는 군집화 방법 중 가장 간단하고 널리 사용되는 군집화 방법

# In[66]:


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3) # k값이 3이기 때문에 n_clusters = 3


# In[67]:


# 군집화 모델에 데이터 적용 
# 지도학습과는 다르게 라벨값 미입력
k_means.fit(train_input)


# In[68]:


KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
      n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
      random_state=None, tol=0.0001, verbose=0)


# In[69]:


# 라벨 속성 확인 
k_means.labels_


# 이는 붓꽃의 라벨을 의미하는 것이 아니라 3개(n_clusters=3)의 군집으로 군집화한 각 군집을 나타낸다.  
# 
# 각 군집의 붓꽃 종의 분포를 확인해보자. 

# In[70]:


print("0 cluster:", train_label[k_means.labels_ ==0])
print("1 cluster:", train_label[k_means.labels_ ==1])
print("2 cluster:", train_label[k_means.labels_ ==2])


# - 결과를 보면 0번째 군집은 라벨 1인 데이터들이 주로 분포
# - 1번째 군집은 라벨 0인 데이터들만 분포
# - 따라서 새로운 데이터에 대해서 0번째 군집으로 예측할 경우 라벨 1로, 1번째 군집으로 예측할 경우 라벨 0으로 2번째 군집으로 예측할 경우 라벨 2로 판단

# 지도 학습과 동일하게 새로운 데이터를 만들어서 예측

# In[71]:


import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])


# 새로 정의한 데이터를 앞서 학습시킨 모델에 적용해 예측값을 확인해보자. 

# In[72]:


prediction = k_means.predict(new_input)
print(prediction)


# - 결과를 보면 새롭게 정의한 데이터는 0번째 군집에 포함된다고 예측.
# - 0번째 군집에는 주로 라벨 1(Versicolour)인 종의 붓꽃들이 군집화  
# 
# 해당 모델의 성능을 측정하기 위해 평가 데이터를 적용시켜서 실제 라벨과 비교해 성능을 측정해보자. 

# In[89]:


predict_cluster = k_means.predict(test_input)
print(predict_cluster)


# 평가 데이터를 적용시커 예측한 군집을 이제 각 붓꽃의 종을 의미하는 라벨값으로 다시 바꿔줘야 실제 라벨과 비교해서 성능을 측정할 수 있다. 

# In[91]:


np_arr = np.array(predict_cluster)
np_arr[np_arr==0], np_arr[np_arr==1], np_arr[np_arr==2] = 3, 4, 5
np_arr[np_arr==3] = 1
np_arr[np_arr==4] = 0
np_arr[np_arr==5] = 2
predict_label = np_arr.tolist()
print(predict_label)


# 0번째 군집이 라벨 1(Versicolour), 1번쨰 군집이 라벨 0(Setosa), 2번째 군집이 라벨 2(Viginica)로 바꿔주었다.  
# 
# 이제 실제 라벨과 비교해서 성능이 어느 정도 되는지 확인해 보자. 

# In[92]:


print('test accuracy {:.2f}'.format(np.mean(predict_label == test_label)))

from sklearn.feature_extraction.text import CountVectorizer


# 데이터 정의 및 객체 생성 
text_data = ['나는 배가 고프다', '내일 점심 뭐먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']

count_vectorizer = CountVectorizer()


count_vectorizer.fit(text_data)
print(count_vectorizer.vocabulary_)



sentence = [text_data[0]] # ['나는 배가 고프다']
print(count_vectorizer.transform(sentence).toarray())


# 모듈 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer


# In[103]:


# 데이터 정의 및 객체 생성
text_data = ['나는 배가 고프다', '내일 점심 뭐먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']

tfidf_vectorizer = TfidfVectorizer()


# In[104]:


tfidf_vectorizer.fit(text_data)
print(tfidf_vectorizer.vocabulary_)

sentence = [text_data[3]] # ['점심 먹고 공부 해야지']
print(tfidf_vectorizer.transform(text_data).toarray())


# '공부'와 '점심'이라는 단어는 0.4 정도의 값을 가지고 '먹고'와 '해야지'는 0.5 정도의 값으로 앞의 두 단어보다 높은 값을 가진다. 이 문장에서 4단어 모두 한 번씩 나왔으나 '먹고'와 '해야지'의 경우 다른 데이터에는 나오지 않은 단어이기 때문에 앞선 두 단어보다 높은 값이 나왔다. 

# ## 자연어 토크나이징 도구 
# - 토크나이징 : 예측해야 할 입력 정보(문장 또는 발화)를 하나의 특정 기본 단위로 자르는 것  

# ### 영어 토크나이징 라이브러리 
# - **NLTK(Natural Language Toolkit)** 와 **Spacy** 라이브러리를 주로 사용 
# 
# **NLTK(Natural Language Toolkit)**
# - 50여 개가 넘는 말뭉치 리소스를 활용해 영어 텍스트를 분석
# - 직관적으로 함수를 쉽게 사용할 수 있게 구성돼 있어 빠르게 텍스트 전처리 가능 

# In[15]:


#라이브러리 설치  
conda install nltk 


# In[18]:


# 말뭉치 설치 함수
import nltk
nltk.download()


# 'all-corpora'는 텍스트 언어 분석을 위한 말뭉치 데이터셋  
# 
# 'all-corpora'만 다운로드!
# 
# ![image.png](attachment:image.png)

# **토크나이징**   
# - 토크나이징이란 텍스트에 대해 특정 기준 단위로 문장을 나누는 것을 의미
# 

# **단어 단위 토크나이징** 
# - 텍스트 데이터를 각 단어를 기준으로 토크나이징

# In[27]:


nltk.download('punkt')


# In[13]:


from nltk.tokenize import word_tokenize
sentence = "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."

print(word_tokenize(sentence))


# ### 문장 단위 토크나이징 
# 경우에 따라 텍스트 데이터를 우선 단어가 아닌 문장으로 나눠야 하는 경우가 있다. 
# 예를 들어, 데이터가 문장으로 구성돼 있어서 문단을 먼저 문장으로 나눈 후 그 결과를 다시 단어로 나눠야하는 경우가 있다. 이런 경우 문장 단위의 토크나이징이 필요하다. 

# In[51]:


from nltk.tokenize import sent_tokenize
paragraph = "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, netural language understanding, and natural language generation."

print(sent_tokenize(paragraph)) 


# NLTK 라이브러리의 경우 토크나이징 외에도 자연어 처리에 유용한 기능들을 제공한다.  
# 대표적으로 데이터를 전처리할 때 경우에 따라 불용어를 제거해야 할 때가 있다. 여기서 불용어란 큰 의미를 가지지 않는 단어를 의미한다. 예를 들어, 영어에서는 'a', 'the'같은 관사나 'is'와 같이 자주 출현하는 단어들을 불용어라 한다.  
# 
# NLTK는 불용어 사전이 내장돼 있어서 따로 불용어를 정의할 필요 없이 바로 사용할 수 있다. 

# ### Spacy
# - NLTK와 같은 오픈소르 라이브러리로 주로 교육, 연구 목적이 아닌 상업용 목적으로 만들어졌다는 점에서 NLTK와 다름  
# - 딥러닝 언어 모델의 개발도 지원

# In[31]:


conda install -c conda-forge spacy


# In[13]:


conda install spacy


# NLTK와 마찬가지로 영어에 대한 텍스트를 전처리하려면 언어 데이터 자료를 별도로 내려받아야 한다. 

# In[49]:


get_ipython().system('python -m spacy download en')


# **Spacy 토크나이징** 
# - Spacy에선 단어 단위와 문장 단위의 토크나이징으로 동일한 모듈을 통해 토크나이징한다. 

# In[52]:


import spacy


# In[53]:


nlp = spacy.load('en')
sentence = "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."

doc = nlp(sentence)


# In[54]:


word_tokenized_sentence = [token.text for token in doc]
sentence_tokenized_list = [sent.text for sent in doc.sents]

print(word_tokenized_sentence)
print(sentence_tokenized_list)


# 토크나이징할 때는 doc 객체를 활용해 [token.text for token in doc]과 같이 리스트 컴프리핸션을 활용하면 간단하게 토크나이징 결과를 확인할 수 있다.   
# 
# **리스트 컴프리핸션**은 파이썬에서 제공하는 기능으로 한 리스트의 모든 원소 각각에 어떤 함수를 적용한 후, 그 반환값을 원소로 가지는 다른 리스트를 쉽게 만들 수 있다.  
# 
# 지금까지 영어 토크나이징 도구에 대해서 알아보았다. 이 도구들은 한국어에는 적용할 수 없다는 단점이 있다.  
# 이제 한글 텍스트 데이터를 어떻게 토크나이징하는지 알아보자. 

# ### 한글 토크나이징 라이브러리

# **KoNLPy**  
# KoNLPy는 한글 자연어 처리를 쉽고 간결하게 처리할 수 있도록 만들어진 오픈소스 라이브러리다.  
# 또한 국내에 이미 만들어져 사용되고 있는 여러 형태소 분석기를 사용할 수 있게 허용한다.  
# 
# 일반적인 어절 단위에 대한 토크나이징은 NLTK로 충분히 해결할 수 있으므로 여기서는 형태소 단위에 대한 토크나이징에 대해 알아보겠다. 

# **윈도우에서 설치**  
# KoNLPy의 경우 기존에 자바로 쓰여진 형태소 분석기를 사용하기 때문에 윈도우에서 KoNLPy를 설치하기 위해서는 1.7이상 버전의 자바가 설치돼 있어야 한다. 
# 
# 커맨드 창에 
# ```
# java -version
# ```
# 
# 라고 쳐서 자바가 설치되어있는지 확인해주면 된다.  
# 
# 이제 환경변수 설정을 해주자  
# 자바 환경설정 하는 방법은 유튜브 영상을 첨부하겠다.   
# https://www.youtube.com/watch?v=GRXhbbs6Go0  
# 
# 환경설정이 끝나면 0.5.7 버전 이상의 JPype1을 설치해야 한다. JPype1은 KoNLPy에서 필요하며 파이썬에서 자바 클래스를 사용할 수 있도록 만들어주는 라이브러리다. 

# In[57]:


pip install JPype1


# In[58]:


pip install konlpy


# In[1]:


import konlpy


# 여기까지 별다른 오류 없이 따라왔다면 제대로 설치된 것이다.   
# 
# 이제 KoNLPy 사용법을 알아보자. 

# **형태소 단위 토크나이징**  
# 한글 텍스트의 경우 형태소 단위 토크나이징이 필요할 때가 있다. 
# KoNLPy의 형태소 분석기는 클래스 형태로 돼 있고, 이를 객체로 생성한 후 메서드를 호출해서 토크나이징할 수 있다. 

# **형태소 분석 및 품사 태깅**  
# - 형태소란 의미를 가지는 가장 작은 단위로 더 쪼개지면 의미를 상실하는 것들을 말함. 
# - 따라서 형태소 분석이란 의미를 가지는 단위를 기준으로 문장을 살펴보는 것을 의미  
# 
# KoNLPy의 형태소 분석기 목록은 다음과 같다.  
# - Hannanum
# - Kkma 
# - Komoran
# - Mecab (윈도우에서 사용 불가) 
# - Okt(Twitter)
# 
# 우리는 Okt를 사용해보자. 

# In[2]:


from konlpy.tag import Okt


# In[3]:


okt = Okt() #okt 객체생성


# Okt는 다음과 같은 4개의 함수를 제공한다.  
# - okt.morphs() 
# : 텍스트를 형태소 단위로 나눈다. 
#     + 옵션으로는 norm과 stem이 있다. 각각 True 혹은 False 값을 받으며, norm은 normalize의 약자로서 문장을 정규화하는 역할을 하고, stem은 각 단어에서 어간을 추출하는 기능이다. 각각 True로 설정하면 각 기능이 적용된다. 옵션을 지정하지 않으면 기본값은 둘 다 False로 설정된다. 
# - okt.nouns()
# : 텍스트에서 명사만 뽑아낸다. 
# - okt.phrases()
# : 텍스트에서 어절을 뽑아낸다. 
# - okt.pos()
# : 각 품사를 태깅하는 역할
#     + 품사를 태깅한다는 것은 텍스트를 형태소 단위로 나누고, 나눠진 각 형태소를 그에 해당하는 품사와 함께 리스트화 하는 것을 의미, 옵션으로 Join함수가 있는데 이 옵션 값을 True로 설정하면 나눠진 형태소와 품사를 '형태소/품사' 형태로 같이 붙여서 리스트화한다. 
# 
# 임의 문장을 직접 지정하고 해당 문장에 각 함수를 직접 적용해 보자. 

# In[5]:


text = "한글 자연어 처리는 재밌다. 이제부터 열심히 해야지ㅎㅎㅎ"

print(okt.morphs(text))
print(okt.morphs(text, stem=True)) #형태소 단위로 나눈 후 어간을 추출


# 어간 추출을 한 경우 '해야지'의 어간인 '하다'로 추출된 것을 볼 수 있다. 이제 앞서 정의한 문장에서 명사와 어절을 추출해 보자. 

# In[6]:


print(okt.nouns(text)) #명사만 추출 
print(okt.phrases(text)) #어절 단위로 나뉘어서 추출 


# 이제 품사를 태깅하는 함수는 pos 함수를 사용해보자 

# In[8]:


print(okt.pos(text))
print(okt.pos(text, join=True)) # 형태소와 품사를 붙여서 리스트화


# **KoNLPy 데이터**  
# KoNLPy 라이브러리는 한글 자연어 처리에 활용할 수 있는 한글 데이터를 포함하고 있다. 따라서 라이브러리를 통해 데이터를 바로 사용할 수 있으며, 데이터의 종류는 다음과 같다.  
# - kolaw 
# : 한국 법률 말뭉치. 'constitution.txt' 파일로 저장돼 있다. 
# - kobill
# : 대한민국 국회 의안 말뭉치. 각 id 값을 가지는 의안으로 구성돼 있고 파일은 '1809890.txt'부터 '1809899.txt'까지로 구성돼 있다.    
# 
# 이제 라이브러리를 사용해 각 데이터를 불러오자. 우선은 각 말뭉치를 불러와야 한다. 

# In[9]:


from konlpy.corpus import kolaw
from konlpy.corpus import kobill


# In[10]:


kolaw.open('constitution.txt').read()[:20] # 법률 말뭉치 불러오기, 긴 말뭉치이므로 앞의 20개까지만 불러오자


# In[12]:


kobill.open('1809890.txt').read()


# 위 데이터들을 여러 가지 한글 자연어 처리 문제를 연습하는 데 활용할 수 있다. 
