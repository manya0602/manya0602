#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris =pd.read_csv("iris.csv")
iris.head()


# In[3]:


iris


# In[4]:


iris.isnull().sum()


# In[5]:


iris.shape


# In[6]:


iris.drop("Id",axis=1,inplace=True)


# In[7]:


iris.head()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()        
iris["Species"]=le.fit_transform(iris["Species"])


# In[9]:


iris


# In[10]:


sns.heatmap(iris.corr(),annot=True,fmt="0.1f")


# In[11]:


X= iris.iloc[:,:-1]
Y=iris.iloc[:,-1]


# In[12]:


Y.head()


# In[13]:


from sklearn.model_selection import train_test_split
x1,x2,y1,y2=train_test_split(X,Y,test_size=0.2,random_state=2, stratify=Y)


# In[14]:


X.head()


# In[15]:


sns.scatterplot(data=iris,x="PetalLengthCm",y="PetalWidthCm",hue="Species")


# In[16]:


sns.scatterplot(data=iris,x="SepalLengthCm",y="SepalWidthCm",hue="Species")


# In[17]:


from sklearn.svm import SVC
svm_model=SVC()
svm_model.fit(x1,y1)


# In[18]:


from sklearn.model_selection import train_test_split
x1,x2,y1,y2=train_test_split(X,Y,test_size=0.2,random_state=2, stratify=Y)


# In[19]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x1,y1)


# In[20]:


model.score(x2,y2)*100


# In[21]:


svm_model.score(x2,y2)*100


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()
knn_model.fit(x2,y2)


# In[23]:


knn_model.score(x2,y2)*100


# In[24]:


from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(x1,y1)


# In[25]:


dt_model.score(x2,y2)*100


# In[ ]:




