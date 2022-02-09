#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
import text2emotion as te

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


# read csv
df = pd.read_csv(r"C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\data\philosophy_data.csv")
df.dropna(inplace=True)


# In[ ]:


# define a function to define the sentiment analysis result 
def senan(c):
    if c>0:
        a = "pos"
    elif c<0:
        a = "neg"
    else:
        a ="neu"
    return a   


# In[ ]:


# use sentiment analysis processing data
Sid = sid()
df['scores'] = df['sentence_spacy'].apply(lambda sentence_spacy: Sid.polarity_scores(sentence_spacy))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(senan)
df.to_csv(r'C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\output\data_new')
df


# In[ ]:


# extract authors name
authors = pd.Series(df['author']).unique()

# plot a bar graph with authors and sentment analysis result
df2=pd.DataFrame(np.zeros((36, 3),int),index=pd.Series(df['author']).unique(),columns=pd.Index(['pos', 'neg',"neu"]))
for senti in ["pos", 'neg',"neu"]:
    for name in authors:
        tempdf = df[df['comp_score']==senti]
        tempdf = tempdf[tempdf["author"]==name]
        num = len(tempdf)
        df2[senti][name]=num 
        
df2.plot.bar()


# In[ ]:


# extract schools name
school = pd.Series(df['school']).unique()

# plot a bar graph with schools and sentment analysis result
df3=pd.DataFrame(np.zeros((13, 3),int),index=school,columns=pd.Index(['pos', 'neg',"neu"]))
for senti in ["pos", 'neg',"neu"]:
    for name in school:
        tempdf2 = df[df['comp_score']==senti]
        tempdf2 = tempdf2[tempdf2["school"]==name]
        num2 = len(tempdf2)
        df3[senti][name]=num2 
df3.plot.bar()
plt.savefig(r'C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\figs\authors_vs_sentment_result.png')


# In[ ]:


# to see authors belong to which schools
df.groupby("school")["author"].unique()


# In[ ]:


# to show the authors in continental group
df4 = df[df["school"] == 'continental']
cont = pd.Series(df4['author']).unique()
cont


# In[ ]:


# similar bar graph with only continental authors included
df5 = pd.DataFrame(np.zeros((3, 3),int),index=cont,columns=pd.Index(['pos', 'neg',"neu"]))
for senti in ["pos", 'neg',"neu"]:
    for name in cont:
        tempdf3 = df[df['comp_score']==senti]
        tempdf3 = tempdf3[tempdf3["author"]==name]
        num3 = len(tempdf3)
        df5[senti][name]=num3 
df5.plot.bar()
plt.savefig(r'C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\figs\cont_vs_sentment_result.png')


# In[ ]:


# define wordcloud plot function
def plot_wordcloud(text,title):
    wordcloud = WordCloud(background_color="white",
                        stopwords = set(STOPWORDS),
                        max_words = 100,
                         max_font_size = 40,
                         relative_scaling = 1,
                         random_state = 3
    ).generate(str(text))
    fig = plt.figure(figsize = (8,8))
    plt.axis('off')
    plt.title(title)
    plt.imshow(wordcloud)


# In[ ]:


# plot wordcloud for continental authors
for i in cont:
    could_0 = df.loc[df['author'] == i]['sentence_lowered']
    plot_wordcloud(could_0,i)
    name = r'C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\figs\wordcloud_' + str(i) + '.png'
    plt.savefig(name)
plt.show()


# In[ ]:


# plot wordcloud for authors in nietzsche school and in feminism school
for i in ["nietzsche","feminism"]:
    df_6 =df.loc[df["school"]==i]
    author1 = pd.Series(df_6["author"]).unique()
    for j in author1:
        could_0 = df.loc[df['author'] == j]['sentence_lowered']
        plot_wordcloud(could_0,j)
        name = r'C:\Users\94333\Python\Github\spring-2022-prj1-Hayinn\figs\wordcloud_' + str(j) + '.png'
        plt.savefig(name)

