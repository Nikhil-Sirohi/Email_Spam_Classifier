#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[49]:


df=pd.read_csv('emails.csv')
df.head(10)


# In[50]:


spam=df[df['spam']==1]


# In[51]:


normal=df[df['spam']==0]


# In[52]:


df.info()


# In[53]:


print('spam_percentage=',(len(spam)/len(df))*100,'%')
print('spam_percentage=',(len(normal)/len(df))*100,'%')


# In[54]:


sns.countplot(df['spam'])


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
encoded_df=vectorizer.fit_transform(df['text'])
print(encoded_df.toarray())


# In[56]:


#print(vectorizer.get_feature_names())


# In[57]:


encoded_df.shape


# In[58]:


label=df['spam'].values
label


# In[59]:


from sklearn.naive_bayes import MultinomialNB
NB_Classifier=MultinomialNB()
NB_Classifier.fit(encoded_df,label)


# In[60]:


testing_sample=["hello Sandeep sir want to earn money","hello Sandeep sir please help!!"]
testing_sample_vectorizer=vectorizer.transform(testing_sample)
testing_sample_predict=NB_Classifier.predict(testing_sample_vectorizer)
testing_sample_predict


# In[61]:


x=encoded_df
y=label
x.shape
y.shape


# In[62]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
NB_Classifier.fit(x_train,y_train)


# In[63]:


from sklearn.metrics import confusion_matrix,classification_report


# In[66]:


y_predict_train=NB_Classifier.predict(x_train)
y_predict_test=NB_Classifier.predict(x_test)


# In[67]:


cm=confusion_matrix(y_predict_train,y_train)
sns.heatmap(cm,annot=True)


# In[68]:


cm=confusion_matrix(y_predict_train,y_train)
sns.heatmap(cm,annot=True)


# In[69]:


print(classification_report(y_test,y_predict_test))


# In[70]:


from sklearn import metrics
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict_test))


# In[ ]:




