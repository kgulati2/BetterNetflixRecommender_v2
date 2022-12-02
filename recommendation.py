# ref: https://www.kaggle.com/code/gazu468/movie-recommendation-system-with-basic-concept#Feature-Extraction-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading Data
movies=pd.read_csv("./data/titles.csv")

#Formatting
movies=movies[['id','title','genres','description', "imdb_id","imdb_score","imdb_votes","tmdb_popularity","tmdb_score"]]

#Cleaning Data
print(movies.isnull().sum())
movies.dropna(inplace=True)
print(movies.isnull().sum())


# genre: convert string representation of list to actual list
def convert(s):
    l = ast.literal_eval(s)
    l = [n.strip() for n in l]
    return l
movies['genres'] = movies['genres'].apply(convert)

# description: convert string to list of words
movies['description']=movies['description'].apply(lambda x: x.split())

# remove space
movies['genres']=movies['genres'].apply(lambda x : [i.replace(" ","")for i in x])
movies['description']=movies['description'].apply(lambda x : [i.replace(" ","")for i in x])

print(movies.head())

movies['tags']= movies['genres'] + movies['description']
partial_df = movies[["id", "title", "tags"]]

partial_df['tags']=partial_df['tags'].apply(lambda x: " ".join(x))
partial_df['tags']=partial_df['tags'].apply(lambda x: x.lower())

nltk.download('wordnet')
nltk.download('omw-1.4')
ps=PorterStemmer()
wnl = WordNetLemmatizer()

def steam(text):
    y=[]
    for i in text.split():
        y.append (ps.stem(i))
    return " ".join(y)

def lemmi(text):
    y=[]
    for i in text.split():
        y.append (wnl.lemmatize(i))
    return " ".join(y)


#lemmitization
partial_df['tags']=partial_df['tags'].apply(lemmi)
#porter_steammer
partial_df['tags']=partial_df['tags'].apply(steam)

tfidf=TfidfVectorizer(max_features=5000,analyzer='word',stop_words="english")
tfdf_features=tfidf.fit_transform(partial_df['tags'])

'''
recommend movie based on input string. This function uses 
tfidf with genre and descriptions of movies in the dataset
as ranking function
@param ipt_str: input string
@return: a list of recommended movie
'''
def recommend_movie(ipt_str):
    rec_list = []
    tfidf_ipt = tfidf.transform([ipt_str])
    tfdf_similarity=cosine_similarity(tfdf_features, tfidf_ipt)
    print(tfdf_similarity)
    movies_list=sorted(list(enumerate(tfdf_similarity)), reverse=True, key=lambda x : x[1])[1:6]
    for i in movies_list:
        rec_list.append(partial_df.iloc[i[0]].title)
        print(partial_df.iloc[i[0]].title)

    return rec_list


#Testing
print(partial_df[partial_df['title']=="Beyond Stranger Things"]["tags"].item())
recommend_movie("stranger thing")
