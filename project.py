#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
import os
from sqlalchemy import create_engine
import datetime as dt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import mlknn
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from datetime import datetime


# In[2]:


os.remove('train.db')
if not os.path.isfile('train.db'):
    start = datetime.now()
    disk_engine = create_engine('sqlite:///train.db')
    start = dt.datetime.now()
    chunksize = 100000
    #j = 0
    #index_start = 1
    path = os.path.join('C:' + os.sep, 'Users', 'ASUS', 'Desktop', 'Train.csv')
    train_df = pd.read_csv(path,names = ['Id','Title','Body','Tags'],chunksize = chunksize)
    #for df in pd.read_csv(path,names = ['Id','Title','Body','Tags'],chunksize = chunksize,iterator = True,encoding = 'utf-8'):
    #    df.index += index_start
    #   j += 1
    #    print('{}rows'.format(j*chunksize))
    df = pd.DataFrame(train_df.get_chunk(chunksize))
    df.to_sql('data',disk_engine,if_exists='append')
    #    index_start = df.index[-1]+1
print("Time taken to run this cell:",datetime.now()-start) 


# In[3]:


if os.path.isfile('train.db'):
    start = datetime.now()
    con = sqlite3.connect('train.db')
    num_rows = pd.read_sql_query("""SELECT count(*) from data""",con)
    print("Number of rows in the database:",num_rows['count(*)'].values[0])
    con.close()
    print("Time taken to count the number of rows:",datetime.now()-start)
else:
    print("Please download the train.db file")


# In[4]:


if os.path.isfile('train.db'):
    start = datetime.now()
    con = sqlite3.connect('train.db')
    df_no_dup = pd.read_sql_query('SELECT Title, Body, Tags, COUNT(*) as cnt_dup FROM data GROUP BY Title,Body, Tags', con)
    con.close()
    print("Time taken to run this cell :", datetime.now() - start)
else:
    print("Please download the train.db file from drive or run the first to genarate train.db file")


# In[5]:


print(df_no_dup)


# In[6]:


df_no_dup.head()
print("number of duplicate questions :", num_rows['count(*)'].values[0]- df_no_dup.shape[0], "(",(1-((df_no_dup.shape[0])/(num_rows['count(*)'].values[0])))*100,"% )")
df_no_dup.cnt_dup.value_counts()


# In[7]:


start = datetime.now()
df_no_dup["tag_count"] = df_no_dup["Tags"].apply(lambda text:len(str(text).split(" ")))
print("Time taken to count the number of rows:",datetime.now()-start)
df_no_dup.head()

df_no_dup.tag_count.value_counts()


# In[8]:


#os.remove('train_no_dupl')

disk_dup = create_engine("sqlite:///train_no_dup.db")
#cursor = disk_dup.cursor()
#cursor.execute("DROP TABLE train_no_du")
no_dup = pd.DataFrame(df_no_dup,columns = ['Title','Body','Tags'])
no_dup.to_sql('train_no_d',disk_dup,if_exists='replace')


# In[9]:


print(no_dup)


# In[10]:


con = sqlite3.connect('train_no_dup.db')
tag_data = pd.read_sql_query("""SELECT Tags FROM train_no_d""",con)
con.close()
tag_data.drop(tag_data.index[0],inplace=True)
tag_data


# In[11]:


vectorizer = CountVectorizer(tokenizer = lambda x:x.split())
tag_dtm = vectorizer.fit_transform(tag_data['Tags'])
print("Number of datapoints:",tag_dtm.shape[0])
print("Number of unique tags:",tag_dtm.shape[1])
tags = vectorizer.get_feature_names()
print("Some of the tags we have:",tags[:10])


# In[12]:


freqs = tag_dtm.sum(axis=0).A1
result = dict(zip(tags,freqs))
if not os.path.isfile('tag_counts_dict_dtm.csv'):
    with open('tag_counts_dict_dtm.csv','w') as csv_file:
        writer = csv.writer(csv_file)
        for key,value in result.items():
            writer.writerow([key,value])
tag_df = pd.read_csv("tag_counts_dict_dtm.csv",names=['Tags','counts'])
tag_df.head()

tag_df_sorted = tag_df.sort_values(['counts'],ascending=False)
tag_counts = tag_df_sorted['counts'].values


plt.plot(tag_counts)
plt.title("Distribution of number of times tag appeared in questions")
plt.grid()
plt.xlabel("Tag Number")
plt.ylabel("Number of time tag appeared")
plt.show()


# In[13]:


plt.plot(tag_counts[0:10000])
plt.title("Distribution of number of times tag appeared in questions")
plt.grid()
plt.xlabel("Tag Number")
plt.ylabel("Number of time tag appeared")
plt.show()


# In[14]:


start = datetime.now()
tup = dict(result.items())
wordcloud = WordCloud(background_color = 'black',
                     width = 16800,
                     height = 1800,
                     ).generate_from_frequencies(tup)
fig = plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
fig.savefig("tag.png")
plt.show()
print("Time taken to run this cell:",datetime.now()-start)


# In[15]:


def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',str(data))
    return cleantext
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


# In[16]:


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


# In[17]:


def create_table(conn,create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
    


# In[18]:


def checkTableExists(dbcon):
    cursr = dbcon.cursor()
    str = "SELECT name from sqlite_master where type = 'table'"
    table_names = cursr.execute(str)
    print("Tables in the database:")
    tables = table_names.fetchall()
    print(tables[0][0])
    return (len(tables))


# In[19]:


def create_database_table(database, query):
    conn = create_connection(database)
    if conn is not None:
        create_table(conn, query)
        checkTableExists(conn)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

sql_create_table = """CREATE TABLE IF NOT EXISTS QuestionsProcessed (question text NOT NULL, code text, tags text, words_pre integer, words_post integer, is_code integer);"""
create_database_table("Processed.db", sql_create_table)
#connectx = sqlite3.connect('Processed.db')


# In[20]:


#res = connectx.execute("SELECT name From sqlite_master Where type='table';")


# In[ ]:





# In[21]:


start = datetime.now()
read_db = 'train_no_dup.db'
write_db = 'Processed.db'
if os.path.isfile(read_db):
    conn_r = create_connection(read_db)
    if conn_r is not None:
        reader =conn_r.cursor()
        reader.execute("SELECT Title, Body, Tags From train_no_d ORDER BY RANDOM() LIMIT 100000;")

if os.path.isfile(write_db):
    conn_w = create_connection(write_db)
    if conn_w is not None:
        tables = checkTableExists(conn_w)
        writer =conn_w.cursor()
        if tables != 0:
            writer.execute("DELETE FROM QuestionsProcessed WHERE 1")
            print("Cleared All the rows")
print("Time taken to run this cell :", datetime.now() - start)


# In[22]:


print(reader.fetchone())
#print(row[1])


# In[23]:


start = datetime.now()
preprocessed_data_list=[]
reader.fetchone()
questions_with_code=0
len_pre=0
len_post=0
questions_proccesed = 0
for row in reader:

    is_code = 0

    title, question, tags = row[0], row[1], row[2]
    
    if '<code>' in question:
        questions_with_code+=1
        is_code = 1
    x = len(title)
    len_pre+=x

    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))

    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
    question=striphtml(question.encode('utf-8'))

    title=title.encode('utf-8')

    question=str(title)
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    len_post+=len(question)
    tup = (question,code,tags,x,len(question),is_code)
    questions_proccesed += 1
    writer.execute("insert into QuestionsProcessed(question,code,tags,words_pre,words_post,is_code) values (?,?,?,?,?,?)",tup)
    if (questions_proccesed%100000==0):
        print("number of questions completed=",questions_proccesed)

no_dup_avg_len_pre=(len_pre*1.0)/questions_proccesed
no_dup_avg_len_post=(len_post*1.0)/questions_proccesed

print( "Avg. length of questions(Title+Body) before processing: %d"%no_dup_avg_len_pre)
print( "Avg. length of questions(Title+Body) after processing: %d"%no_dup_avg_len_post)
print ("Percent of questions containing code: %d"%((questions_with_code*100.0)/questions_proccesed))

print("Time taken to run this cell :", datetime.now() - start)
conn_r.commit()
conn_w.commit()
conn_r.close()
conn_w.close()


# In[24]:


if os.path.isfile(write_db):
    conn_r = create_connection(write_db)
    if conn_r is not None:
        reader =conn_r.cursor()
        reader.execute("SELECT question From QuestionsProcessed LIMIT 10")
        print("Questions after preprocessed")
        print('='*100)
        reader.fetchone()
        for row in reader:
            print(row)
            print('-'*100)
conn_r.commit()
conn_r.close()


# In[25]:


write_db = 'Processed.db'
if os.path.isfile(write_db):
    conn_r = create_connection(write_db)
    if conn_r is not None:
        preprocessed_data = pd.read_sql_query("""SELECT question, Tags FROM QuestionsProcessed""", conn_r)
conn_r.commit()
conn_r.close()


# In[26]:


preprocessed_data.head()
print("number of data points in sample :", preprocessed_data.shape[0])
print("number of dimensions :", preprocessed_data.shape[1])


# In[27]:


vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
multilabel_y = vectorizer.fit_transform(preprocessed_data['tags'])


# In[28]:


def tags_to_choose(n):
    t = multilabel_y.sum(axis=0).tolist()[0]
    sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
    multilabel_yn=multilabel_y[:,sorted_tags_i[:n]]
    return multilabel_yn

def questions_explained_fn(n):
    multilabel_yn = tags_to_choose(n)
    x= multilabel_yn.sum(axis=1)
    return (np.count_nonzero(x==0))


# In[29]:


questions_explained = []
total_tags=multilabel_y.shape[1]
total_qs=preprocessed_data.shape[0]
for i in range(500, total_tags, 100):
    questions_explained.append(np.round(((total_qs-questions_explained_fn(i))/total_qs)*100,3))


# In[37]:


fig, ax = plt.subplots()
ax.plot(questions_explained)
xlabel = list(500+np.array(range(-50,450,50))*50)
ax.set_xticklabels(xlabel)
plt.xlabel("Number of tags")
plt.ylabel("Number Questions coverd partially")
plt.grid()
plt.show()
# choose any number of tags based on the computing power, minimun is 50(it covers 90% of the tags)
print("with ",500,"tags we are covering ",questions_explained[0],"% of questions")


# In[36]:


multilabel_yx = tags_to_choose(500)
print("number of questions that are not covered :", questions_explained_fn(500),"out of ", total_qs)
print("Number of tags in sample :", multilabel_y.shape[1])
print("number of tags taken :", multilabel_yx.shape[1],"(",(multilabel_yx.shape[1]/multilabel_y.shape[1])*100,"%)")


# In[32]:


total_size=preprocessed_data.shape[0]
train_size=int(0.80*total_size)

x_train=preprocessed_data.head(train_size)
x_test=preprocessed_data.tail(total_size - train_size)

y_train = multilabel_yx[0:train_size,:]
y_test = multilabel_yx[train_size:total_size,:]
print("Number of data points in train data :", y_train.shape)
print("Number of data points in test data :", y_test.shape)


# In[33]:


start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))
x_train_multilabel = vectorizer.fit_transform(x_train['question'])
x_test_multilabel = vectorizer.transform(x_test['question'])
print("Time taken to run this cell :", datetime.now() - start)
print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)
print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)


# In[34]:


classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict(x_test_multilabel)

print("accuracy :",metrics.accuracy_score(y_test,predictions))
print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))
print("hamming loss :",metrics.hamming_loss(y_test,predictions))
print("Precision recall report :\n",metrics.classification_report(y_test, predictions))


# In[ ]:




