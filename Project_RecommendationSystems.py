#!/usr/bin/env python
# coding: utf-8

# # OTT Platforms Recommendation Systems 

# ### submitted by: Anurag Sen (as5864@srmist.edu.in) 

# OTT platform recommendation system. It can suggest a new user a series/movie based on the movie weight rate by IMDB. It can perform personalized recommendation based on title similarity. Designed using selection criterion of IMBD and user specific recommendation system using Tfidfvectorizer and Cosine Similarity.
# 
# The program was built in Python:
# 
# 1. IMDB weight rate criterion
# 2. Tfidvectorizer

# In[1]:


#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("C:\\Users\Anurag Sen\Desktop\Recomm Systems\movies_metadata (1).csv")
data.head()


# ### Data Preprocessing

# In[3]:


data['genres'] = data['genres'].fillna('[]')  #filling empty entries


# In[4]:


data.head(100)


# In[5]:


data['genres'] = data['genres'].apply(literal_eval) #converting genres from string to a list


# In[6]:


data.head()


# In[7]:


#Genres as a list and eliminating keys from the dictionary and keeping just the values


# In[8]:


data['genres'] = data['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[9]:


data.head()


# In[10]:


#converting non-null vote_count and vote_average into integer


# In[11]:


votecount = data[data['vote_count'].notnull()]


# In[12]:


vote_count = votecount['vote_count'].astype('int')
vote_count


# In[13]:


voteaverage = data[data['vote_average'].notnull()]


# In[14]:


vote_average = voteaverage['vote_average'].astype('int')
vote_average


# ### Top Movies Recommendation System

# In[15]:


#sorting the average votes in Descending order and number of votes greater than 1000


# In[16]:


top_movies = data.copy()


# #### Sorting the vote_average in descending order 

# In[17]:


top_movies1 = top_movies.sort_values('vote_average', ascending=False).head(250)


# In[18]:


top_movies1


# #### Selecting the vote_count greater than 1000 

# In[19]:


top_movies2 = top_movies[top_movies['vote_count']>1000]


# In[20]:


top_movies2


# #### Vote_average in descending order || Vote_count greater than 1000 

# In[21]:


top_movies2.sort_values('vote_average', ascending=False).head(250) 


# ### IMDB Weighted Rating 

# In[22]:


#Weighted Rating

#W = (Rv + Cm)/(v+m)

#W = weighted rating
#R = Ratings
#v = number of votes
#C = the mean vote accross the whole report
#m = minimum votes required to be in Top250


# In[23]:


C = vote_average.mean()
C


# In[24]:


m = vote_count.quantile(0.95)
m


# In[25]:


top_movies['year'] = pd.to_datetime(top_movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[26]:


top_movies


# In[27]:


top_movies3 = top_movies[(top_movies['vote_count'] >= m) & (top_movies['vote_count'].notnull()) & (top_movies['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity','genres']]
top_movies3.shape


# In[28]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    W = ((R*v) + (C*m))/(m+v)
    return W


# In[29]:


top_movies3['IMDB weight_rate'] = top_movies3.apply(weighted_rating, axis=1)


# In[30]:


top_movies3


# In[31]:


top_movies3 = top_movies3.sort_values('IMDB weight_rate', ascending=False)
top_movies3


# In[32]:


#All the rating adjustments are general algorithmic ratings, BUT NOT USER SPECIFIC*


# ### Genre based division

# In[33]:


genre_TM = top_movies3.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1,drop=True)
genre_TM.name = 'genre'
genre_top_movies = top_movies3.drop('genres',axis=1).join(genre_TM)
genre_top_movies = genre_top_movies.sort_values('IMDB weight_rate', ascending=False)
genre_top_movies.head(10)


# In[34]:


genre_top_movies['genre'].unique()


# In[35]:


genre_top_movies[genre_top_movies.genre == "Animation"]


# In[36]:


genre_top_movies[genre_top_movies.genre == "Family"]


# In[37]:


genre_top_movies[genre_top_movies.genre == "Action"]


# In[38]:


genre_top_movies[genre_top_movies.genre == "Drama"]


# ### Content Based Recommendation

# In[39]:


#Customized Recommendation


# In[40]:


links_small = pd.read_csv("C:\\Users\Anurag Sen\Desktop\Recomm Systems\links_small.csv")
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[41]:


top_movies = top_movies.drop([19730, 29503, 35587])


# In[42]:


top_movies['id'] = top_movies['id'].astype('int')
top_movies4 = top_movies[top_movies['id'].isin(links_small)]
top_movies4.shape


# In[43]:


top_movies4.head()


# In[44]:


top_movies4['tagline'] = top_movies4['tagline'].fillna('')
top_movies4['description'] = top_movies4['overview'] + top_movies4['tagline']
top_movies4['description'] = top_movies4['description'].fillna('na')


# In[45]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
tfidf_matrix = tf.fit_transform(top_movies4['description'])


# In[46]:


tfidf_matrix


# In[47]:


tfidf_matrix.shape


# #### Cosine similarity of words
# ##### It will return similar movies to the given title

# In[48]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[49]:


cosine_sim


# In[50]:


cosine_sim[0]


# Pairwise cosine similarity is noted. Now to make a function that returns 30 most similar movies based on cosine similarity

# In[51]:


top_movies4 = top_movies4.reset_index()
titles = top_movies4['title']
indices = pd.Series(top_movies4.index, index = top_movies4['title'])


# In[52]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[53]:


get_recommendations('The Dark Knight').head(10)


# In[54]:


get_recommendations('The Godfather').head(10)


# In[55]:


get_recommendations('Toy Story').head(10)

