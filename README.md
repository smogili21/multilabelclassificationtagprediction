# multilabelclassificationtagprediction

# Multi-Label Classification using Codeforces Platform 

CodeForce is a Programming language Competitive platform where Questions are posted, usually its quite challenging to come up with an approach for solving looking at the description. So, tagging for questions is essential for a better user experience. In this machine learning problem, the final goal is to predict the tag based on the Question's textual description, abbreviated as 'problem statement'.

<h3>Goal and Objectives:</h3>

1)Conduct Explorary Data Analysis on problem tags to analyse number of tags per question,number of words in tags and number of unigrams and multigrams in the problem statement.

2)Conduct Data Pre-processing on problem statement to convert text data to lower case,remove unicode characters , html tags,stop words removal,lemma and stemming.

3)Run and compare various Machine Learning algorithms(Multi-label Logistic Regression,Multi-label Random Forest Classifier and Bilstm with Embedding layer(DL algorithm))by performing hyper-parameter tuning to calculate precision,recall,f1-score,hamming-loss.



import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#Read the csv to dataframe.
df = pd.read_csv("data.csv")
df.head()

df.info()

<h3>Observation:</h3>
<p>There are total of 5 columns</p>

df.isnull().sum()

#Separate problem difficulty from the problem tags , has prblem difficulty is not considered for tag analysis
s=df['problem_tags'].str.split(',').str[-1]
difficult_values=s.str.isnumeric()
count=0
for val in difficult_values:
    if(val):
        count+=1
        df['problem_difficulty']=df['problem_tags'].str.split(',').str[-1]
        df["problem_difficulty"] = df["problem_difficulty"].apply(lambda x:str(x)[1:])
        df['problem_tags_values']=df['problem_tags'].str.split(',').str[:-1]    
print(count)
df.head()

erow_indices_to_drop = [ x for x in list(range(df.shape[0])) if df["problem_tags_values"].iloc[x] == []]

df["problem_tags_values"].describe()
row_indices_to_drop = [ x for x in list(range(df.shape[0])) if df["problem_tags_values"].iloc[x] == []]

#Drop the empty values with empty  problem tags
df = df.drop(row_indices_to_drop)

df["problem_tags_values"].describe()
df.head()

#Drop the missing values from dataframe
df = df.dropna()

#Calculate number of problem tags in the dataset
tags_dict = {}
for allTags in df["problem_tags_values"]:
    for tag in allTags :
        if tag not in tags_dict:
            tags_dict[tag] = 0 
        tags_dict[tag] += 1 

tags_dict_sorted = dict(sorted(tags_dict.items(), key = lambda x: x[1], reverse = True))
tag_count = {"Tag":list(tags_dict_sorted.keys()), "Count": list(tags_dict_sorted.values())}
count_df = pd.DataFrame(data = tag_count)
count_df[:10]

tags_dict = {}
for allTags in df["problem_tags_values"]:
    for tag in allTags :
        if tag not in tags_dict:
            tags_dict[tag] = 0 
        tags_dict[tag] += 1 

tags_dict_sorted = dict(sorted(tags_dict.items(), key = lambda x: x[1], reverse = True))
tag_count = {"Tag":list(tags_dict_sorted.keys()), "Count": list(tags_dict_sorted.values())}
count_df = pd.DataFrame(data = tag_count)
total_tags=count_df.Tag

axes = count_df.head(20).plot(x = 'Tag', y = 'Count', kind = 'bar', figsize = (18, 10), fontsize = 15, grid = True)
plt.xlabel("")
plt.ylabel("Count", fontsize = 20)
plt.title("Top 20 Highest occurring Tags", fontsize = 20)

<h3> Observations:</h3>
<p> Majority of the most frequent tags are implementation and maths tags</p>

#Converting the dataframe column to string value.
df["problem_tags_values"] = df["problem_tags_values"].apply(lambda x:" ".join(x))

df.isnull().sum()


<h3>Observation:</h3>
<p>From the above we can conclude that null values are removed from the dataframe.</p>

#Drop the column problem_tags as they are segregated into separate columns.
df = df.drop('problem_tags', 1)

# Check if there are duplicate data points in the Problem statement.
df2=df.groupby(df.columns.tolist(),as_index=False).size()

df2

df2.groupby('size').count()

<h3>Observation: </h3>
<p>No Duplicate data points that is denoted by the value in 1.</p>

# Calculate the number of tags per Question which is denoted by 'tags_count'
df['tags_count'] = [len(i.split(" ")) for i in df["problem_tags_values"] ]
df

#Find the maximum, minimum and mean tags per question.
min_tag_count = df["tags_count"].min()
max_tag_count = df["tags_count"].max()
avg_tag_count = df["tags_count"].mean()
meadian_tag_count = df["tags_count"].median()

print("Maximum tags per question:" + str(max_tag_count))
print("Minimum tags per question:" + str(min_tag_count))
print("Mean tags per question:" + str(avg_tag_count))
print("Median tags per question:" + str(meadian_tag_count))

df['tags_count'].value_counts()

   <h3>Observations:</h3>
<ul>1.Maximum number of tags per question: 11 </ul>
<ul>2.Minimum number of tags per question: 1 </ul>
<ul>3.Avg. number of tags per question: 2.561 </ul>
<ul>4.Most of the questions are having 2 or 3 tags </ul>

<h3>Plot to show number of words in the Tags</h3>


tag_word_count=[len(list(tags_dict_sorted.keys())[list(tags_dict_sorted.keys()).index(index)]) for index in list(tags_dict_sorted.keys())]
tags=list(tags_dict_sorted.keys())


d = {'Tag_word_count': tag_word_count, 'Tags': tags}
df_tag_word_count = pd.DataFrame(data=d)
df_tag_word_count

plot_tags=df_tag_word_count.plot(kind="bar",x='Tags', width=0.8)
plt.title('Plot to show number of words in the Tags')
plt.xlabel('Tags')
plt.ylabel('Count of words in the Tags')

<h3>Bi-gram and Multi-gram Analysis for Problem Tags</h3>

count_multigrams=0
count_datapoints=0

tag_value_list=df.problem_tags_values

for i in tag_value_list:
    for j, x in enumerate(i.split()):
        a=i.split()
        count_datapoints+=1
        if(len(a)>1):
            count_multigrams+=1
print("Total tokenized words:",count_datapoints)
print("Number of unigrams in the dataset are:",count_datapoints-count_multigrams)
print("Number of Multigrams in the dataset are:",count_multigrams)

import matplotlib.pyplot as plt
import seaborn as sns

data = [count_datapoints-count_multigrams,count_multigrams]
labels = ['Total_Unigrams', 'Total_Muligrams']

my_explode=[0.2,0]
colors = sns.color_palette('Paired')
         
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%',explode=my_explode)
plt.title('Bi-gram and Multi-gram Analysis for Problem Tags')
plt.show()

#Problem Statement example to show it is not yet pre-processed
print("Problem Statement example to show it is not yet pre-processed:")
df.problem_statement[17]

<h3>Data Preprocessing for the questions</h3>
<p> </p>

import re

df_processed=df

#replace all the \n in the text
problem_stat_processed = df_processed.problem_statement.replace(r'\n',' ', regex=True) 

#convert to lower case
problem_stat_processed=problem_stat_processed.str.lower()

#Remove unicodes 
problem_stat_processed=[re.sub('\s[\$\s]+\w*[\$\s]+', '', item.encode('ascii', 'ignore').decode('ascii')) for item in problem_stat_processed ]
CLEANR = re.compile('<.*?>') 
re_pattern = re.compile(u'[^\u2009-\uD7FF\uE000-\uFFFF]', re.UNICODE)

#html tags removal
problem_stat_processed=[re.sub(CLEANR, ' ', str(item)) for item in problem_stat_processed ]

print("Pre-processed problem statment:")
problem_stat_processed[17]


 import nltk
 nltk.download('punkt')

 import nltk
 nltk.download('stopwords')

 import nltk
 nltk.download('wordnet')

a=problem_stat_processed
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS

#stop words removal.
problem_stat_processed = [" ".join([word for word in word_tokenize(sentence) if word not in STOP_WORDS]) for sentence in a]
                                                        

print("Pre-processed problem statment after stop words removal:")
problem_stat_processed[17]

#Lemmatanization
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

text_problem=problem_stat_processed
problem_stat_processed = [" ".join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence) ]) for sentence in text_problem]

print("Pre-processed problem statment after Lemmatanization:")
problem_stat_processed[17]

#Stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
text_problem=problem_stat_processed
problem_stat_processed = [" ".join([ps.stem(word) for word in word_tokenize(sentence) ]) for sentence in text_problem]

print("Pre-processed problem statment after Stemming:")
problem_stat_processed[17]

#Setting back to dataframe the pre-processing text
df.problem_statement=problem_stat_processed

from sklearn.feature_extraction.text import CountVectorizer

#Converting tags to multi-label
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
multi_tags = vectorizer.fit_transform(df.problem_tags_values)
print("Total number of tags in the dataset:",multi_tags.shape[1])

import re
def clean_data(row):
    return re.sub('\s[\$\s]+\w*[\$\s]+', '', row.encode('ascii', 'ignore').decode('ascii'))

df['problem_statement'] = df['problem_statement'].apply(clean_data)

**Bag of Words**
Featurizing bag of words with text vectorizer

from sklearn.model_selection import train_test_split

X = df[['problem_statement']]
y = df[["problem_tags_values"]]

#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer
#Vectorizing the Inputs.
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
x_train_multilabel = vectorizer.fit_transform(X_train['problem_statement'])
x_test_multilabel = vectorizer.transform(X_test['problem_statement'])

y_train = vectorizer.fit_transform(y_train['problem_tags_values'])
y_test = vectorizer.fit_transform(y_test['problem_tags_values'])


print("Train data diemnsions are :",x_train_multilabel.shape, "Y :",y_train.shape)
print("Test data dimensions are :",x_test_multilabel.shape,"Y:",y_test.shape)

<h3>Logistic Regression with OnevsRestClassifier Model</h3>

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support as score

#LogisticRegression for multi-label classification.
classifier_2 = OneVsRestClassifier(LogisticRegression(penalty='l2',solver='sag', max_iter=1000))
classifier_2.fit(x_train_multilabel, y_train)
predictions_2 = classifier_2.predict(x_test_multilabel)

print("Accuracy of the Logistic Regression is:",metrics.accuracy_score(y_test, predictions_2))
print("Hamming loss of the Logistic Regression is",metrics.hamming_loss(y_test,predictions_2))

precision = precision_score(y_test, predictions_2, average='micro')
recall = recall_score(y_test, predictions_2, average='micro')
f1 = f1_score(y_test, predictions_2, average='micro')
 
print("Micro-average scores")
print("Precision are : {:.4f}, Recall are : {:.4f}, F1-measure are : {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, predictions_2, average='macro')
recall = recall_score(y_test, predictions_2, average='macro')
f1 = f1_score(y_test, predictions_2, average='macro')
 
print("Macro-average scores")
print("Precision are : {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision, recall, f1))

print (metrics.classification_report(y_test, predictions_2))


Hyper parameters tuning for OneVsRestClassifier(Logistic Regression)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Hyper paramter tuning for best accuracy.
param_grid = dict(estimator__C=[1, 1.2], estimator__penalty = ['l1', 'l2'], estimator__solver = ['liblinear', 'sag'])
gsv = GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced')), param_grid=param_grid, verbose=5, n_jobs=-1)
gsv.fit(x_train_multilabel, y_train)

print('Hyper parameters are  ', gsv.best_params_)
print('Best Score values are ', gsv.best_score_)

classifier = OneVsRestClassifier(LogisticRegression(C=1.2, solver='liblinear', penalty='l2'))
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict (x_test_multilabel)

print("Accuracy after hyper-parameter tuning is :",metrics.accuracy_score(y_test, predictions))
print("Hamming loss after hyper-parameter tuning ",metrics.hamming_loss(y_test,predictions))

precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')
f1 = f1_score(y_test, predictions, average='micro')
 
print("Micro-average score after tuning")
print("Precision are: {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
 
print("Macro-average score after tuning")
print("Precision are: {:.4f}, Recall are: {:.4f}, F1-measure  are: {:.4f}".format(precision, recall, f1))

print (metrics.classification_report(y_test, predictions))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Printing multi-label confusion matrix .
def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("CM-" + class_label)

#Logistic-Regression
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

labels=total_tags.astype(str).values.tolist()
cfs_matrix=multilabel_confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(6, 6, figsize=(14, 10))
for axes, cfs_matrix, label in zip(ax.flatten(), cfs_matrix, labels):
  print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"]) 
fig.tight_layout()
plt.show()

**Multi-label Confusion matrix for Logistic Regression**

The above plot shows the confusin matrix for multi-label classification and we can see that false positive values are quite large . 

**SGDClassifier for Multi-label Classification**

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support as score

#Linear SVM
classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=0.00001, penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions_sdg = classifier.predict(x_test_multilabel)

print("Accuracy of the SGD Classifier is:",metrics.accuracy_score(y_test, predictions_sdg))
print("Hamming loss of the SGD Classifier is",metrics.hamming_loss(y_test, predictions_sdg))

precision = precision_score(y_test, predictions_sdg, average='micro')
recall = recall_score(y_test, predictions_sdg, average='micro')
f1 = f1_score(y_test, predictions_sdg, average='micro')
 
print("Micro-average scores")
print("Precision are : {:.4f}, Recall are : {:.4f}, F1-measure are : {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, predictions_sdg, average='macro')
recall = recall_score(y_test, predictions_sdg, average='macro')
f1 = f1_score(y_test, predictions_sdg, average='macro')
 
print("Macro-average scores")
print("Precision are : {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision, recall, f1))

print (metrics.classification_report(y_test, predictions_sdg))

# Hyper paramter tuning for best accuracy.
param_grid = dict(estimator__alpha=[0.001, 0.01], estimator__penalty=['l1', 'l2', 'elasticnet'])
gsv = GridSearchCV(OneVsRestClassifier(SGDClassifier()), param_grid=param_grid, verbose=5, n_jobs=-1)
gsv.fit(x_train_multilabel, y_train)

print('Hyper parameters are  ', gsv.best_params_)
print('Best Score values are ', gsv.best_score_)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support as score

classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=0.001, penalty='l2'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions_sdg = classifier.predict(x_test_multilabel)

print("Accuracy of the SGD Classifier is:",metrics.accuracy_score(y_test, predictions_sdg))
print("Hamming loss of the SGD Classifier is",metrics.hamming_loss(y_test, predictions_sdg))

precision = precision_score(y_test, predictions_sdg, average='micro')
recall = recall_score(y_test, predictions_sdg, average='micro')
f1 = f1_score(y_test, predictions_sdg, average='micro')
 
print("Micro-average scores")
print("Precision are : {:.4f}, Recall are : {:.4f}, F1-measure are : {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, predictions_sdg, average='macro')
recall = recall_score(y_test, predictions_sdg, average='macro')
f1 = f1_score(y_test, predictions_sdg, average='macro')
 
print("Macro-average scores")
print("Precision are : {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision, recall, f1))

print (metrics.classification_report(y_test, predictions_sdg))

**Random Forest Classifier for Multi-label Classification**

from sklearn.ensemble import RandomForestClassifier
rfc_classifier = RandomForestClassifier()

#Random forest classifier model
rfc_clf = OneVsRestClassifier(rfc_classifier)
rfc_clf.fit(x_train_multilabel, y_train)

rfc_predictions = rfc_clf.predict(x_test_multilabel)

print("Accuracy of the Random forest Classifier is:",metrics.accuracy_score(y_test, rfc_predictions))
rf_hamming_loss=metrics.hamming_loss(y_test, rfc_predictions)
print("Hamming loss of the Random forest Classifier is",rf_hamming_loss)

precision_rf = precision_score(y_test, rfc_predictions, average='micro')
recall_rf = recall_score(y_test, rfc_predictions, average='micro')
f1_rf = f1_score(y_test, rfc_predictions, average='micro')
 
print("Micro-average scores")
print("Precision are : {:.4f}, Recall are : {:.4f}, F1-measure are : {:.4f}".format(precision_rf, recall_rf, f1_rf))

precision_rf_macro = precision_score(y_test, rfc_predictions, average='macro')
recall_rf_macro = recall_score(y_test, rfc_predictions, average='macro')
f1_rf_macro = f1_score(y_test, rfc_predictions, average='macro')
 
print("Macro-average scores")
print("Precision are : {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision_rf_macro, recall_rf_macro, f1_rf_macro))

print (metrics.classification_report(y_test, rfc_predictions))

#Random forest Classifier
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

labels=total_tags.astype(str).values.tolist()
cfs_matrix=multilabel_confusion_matrix(y_test, rfc_predictions)
fig, ax = plt.subplots(6, 6, figsize=(14, 10))
for axes, cfs_matrix, label in zip(ax.flatten(), cfs_matrix, labels):
  print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"]) 
fig.tight_layout()
plt.show()

**Bi-LSTM model for multi-label classification**

from tensorflow.keras.preprocessing.text import Tokenizer

#Tokenize the x_train values
title_encoder = Tokenizer(oov_token="<ukn>")
title_encoder.fit_on_texts(X_train['problem_statement'].values)
train_titles = title_encoder.texts_to_sequences(X_train['problem_statement'].values)
test_titles = title_encoder.texts_to_sequences(X_test['problem_statement'].values)

len(title_encoder.word_index)

lens = [len(x) for x in train_titles]
plt.plot(sorted(lens), list(range(len(lens))))

We can see a peak at 250 so we set as padding length.

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_titles = pad_sequences(train_titles,maxlen=250)
test_titles = pad_sequences(test_titles,maxlen=250)

# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding


# get the model
def get_model(n_inputs, n_outputs):

  bilstm = tf.keras.Sequential()
  bilstm.add(tf.keras.layers.Embedding(input_dim = 28525+1,output_dim = 100,input_length=250))
  bilstm.add(tf.keras.layers.Bidirectional(LSTM(100, dropout=0.25,return_sequences=False)))
  bilstm.add(tf.keras.layers.Dense(20,input_dim=n_inputs, activation='relu'))
  bilstm.add(tf.keras.layers.Dense(37, activation='sigmoid'))
  bilstm.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
  return bilstm

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y,n_input,n_output):
  results=list()
  model = get_model(n_input, n_output)
  model.fit(X, y, verbose=1, epochs=100, batch_size=10)
  yhat=model.predict(test_titles)
  return yhat


n_input=x_train_multilabel.shape[1]
n_output=y_train.shape[1]
train_x = x_train_multilabel.toarray()
train_y = y_train.toarray()
validation_x = x_test_multilabel.toarray()
validation_y = y_test.toarray()

# evaluate model
yhat=evaluate_model(train_titles, y_train.toarray(),n_input,n_output)
yhat[yhat>0.5] = 1
yhat[yhat<=0.5] = 0
print(metrics.classification_report(y_test, yhat))

 # mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding


# get the model
def get_model(n_inputs, n_outputs):

  bilstm = tf.keras.Sequential()
  bilstm.add(tf.keras.layers.Embedding(input_dim = 28525+1,output_dim = 100,input_length=250))
  bilstm.add(tf.keras.layers.Bidirectional(LSTM(100, dropout=0.25,return_sequences=False)))
  bilstm.add(tf.keras.layers.Dense(20,input_dim=n_inputs, activation='relu'))
  bilstm.add(tf.keras.layers.Dense(37, activation='sigmoid'))
  bilstm.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
  return bilstm

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y,n_input,n_output):
  results=list()
  model = get_model(n_input, n_output)
  model.fit(X, y, verbose=1, epochs=50, batch_size=1)
  yhat=model.predict(test_titles)
  return yhat


n_input=x_train_multilabel.shape[1]
n_output=y_train.shape[1]
train_x = x_train_multilabel.toarray()
train_y = y_train.toarray()
validation_x = x_test_multilabel.toarray()
validation_y = y_test.toarray()

# evaluate model
yhat=evaluate_model(train_titles, y_train.toarray(),n_input,n_output)
yhat[yhat>0.5] = 1
yhat[yhat<=0.5] = 0
print(metrics.classification_report(y_test, yhat))


#Classification and Performance Metric Calculation.
print("Accuracy of the Bilstm is:",metrics.accuracy_score(y_test, yhat))
print("Hamming loss of the Bilstm is",metrics.hamming_loss(y_test,yhat))

precision_bilstm = precision_score(y_test, yhat, average='micro')
recall_bilstm = recall_score(y_test, yhat, average='micro')
f1_bisltm = f1_score(y_test, yhat, average='micro')
hamming_bilstm=metrics.hamming_loss(y_test,yhat)
 
print("Micro-average scores")
print("Precision are : {:.4f}, Recall are : {:.4f}, F1-measure are : {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, yhat, average='macro')
recall = recall_score(y_test, yhat, average='macro')
f1 = f1_score(y_test, yhat, average='macro')
 
print("Macro-average scores")
print("Precision are : {:.4f}, Recall are: {:.4f}, F1-measure are: {:.4f}".format(precision, recall, f1))

print (metrics.classification_report(y_test, yhat))

**Observation**
Bilstm model gave a f1 score 0.40 even with batch_size=1 and epochs=50. This is due to overfitting.


**Future Work**

For better f1 score values we can implement the SentenceTransformers as a future work. It works well with image and text embeddings

**Performance Metrics**

 from tabulate import tabulate

 print("Performance Metrics\n")
 d = [[1,"Logistic-Regression", round(precision_score(y_test, predictions, average='micro'),3), round(recall_score(y_test, predictions, average='micro'),3),round(f1_score(y_test, predictions, average='micro'),3),round(metrics.hamming_loss(y_test,predictions),3)],
     [2,"Random Forest Classifier", round(precision_rf ,3), round(recall_rf ,3),round(f1_rf,3),round(rf_hamming_loss,3)],
     [3,"Bidirectional-lstm",round(precision_bilstm ,3),round(recall_bilstm ,3),round(f1_bisltm ,3),round(hamming_bilstm,3)],
     [4,"SGDClassifier",round(precision_score(y_test, predictions_sdg, average='micro'),3), round(recall_score(y_test, predictions_sdg, average='micro'),3),round(f1_score(y_test, predictions_sdg, average='micro'),3),round(metrics.hamming_loss(y_test,predictions_sdg),3)]]
     
print(tabulate(d, headers=['Sl-no','Algorithm','Precision','Recall','Micro-f1score','Hamming-loss']))

**Model Performance Conclusion**

From the above table we can see that Logistic Regression has the best f1-score compared to random forest regressor and lstm . This may be due to small dataset size and also overfitting. Techniques like oversampling or undersampling can be applied to make the dataset with various tags more balanced. As the dataset is small , machine learning model is performing better than deep learning model. 

However, Bert can be used to improve the f1-score further.


