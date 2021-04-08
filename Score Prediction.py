#!/usr/bin/env python
# coding: utf-8

# # PREDICTION USING SUPERVISED ML

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('student_scores - student_scores.csv')


# In[3]:


data.head(8)


# In[4]:


data.info()


# In[5]:


data.columns


# In[6]:


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[7]:


X


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=21)


# In[11]:


X_train.shape


# In[12]:


y_train.shape


# In[13]:


X_test.shape


# In[14]:


y_test.shape


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


model = LinearRegression() 


# In[17]:


model.fit(X_train, y_train)


# In[18]:


X_test


# In[19]:


predicted_values = model.predict(X_test)


# In[20]:


predicted_values


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


predicted_values.shape


# In[23]:


y_test.shape


# In[24]:


accuracy = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_values})


# In[25]:


accuracy


# In[26]:


data.plot(x="Hours", y="Scores")
plt.title("HOURS vs SCORES")
plt.xlabel("HOURS")
plt.ylabel("SCORES")


# In[27]:


hours = [[9.25]]
prediction = model.predict(hours)
prediction

