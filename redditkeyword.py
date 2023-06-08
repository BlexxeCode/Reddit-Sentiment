import string
import plotly.express as px
import requests
import praw
from config import reddit
from PIL import Image
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from IPython.display import display
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sentiment import Full_process
from ml_models import nltk_sentiment, spacy_sentiment
from nltk.stem import WordNetLemmatizer


def get_sub_info(search, sub):
    if sub == "":
        subreddit = reddit.subreddit("all").search(search, sort='hot')
    else:
        subreddit = reddit.subreddit(sub).search(search, sort='hot')
    ids = []
    for submission in subreddit:
        ids.append(submission.id)

    titles = []
    bodies = []
    first_level_replies = []
    second_level_replies = []
    all_post = []

    for i,id in enumerate(ids):
        post = reddit.submission(id = id)
        titles.append(post.title)
        all_post.append([post.title])
        bodies.append(post.selftext)
        all_post[i].append(post.selftext)
        post.comment_sort = 'top'
        post.comments.replace_more()
        for first_level_reply in post.comments:
            first_level_replies.append(first_level_reply.body)
            all_post[i].append(first_level_reply.body)
            for second_level_reply in first_level_reply.replies:
                second_level_replies.append(second_level_reply.body)
                all_post[i].append(second_level_reply.body)

    print(all_post)
    for title, body, replyone, replytwo in zip(titles, bodies, first_level_replies, second_level_replies):
        print(title)
        print(body)
        print('-'*10)
    all_text = titles + bodies + first_level_replies + second_level_replies
    return all_post, all_text

def get_wordcount(text):
    stop_words = list(stopwords.words('english'))
    stop_words.append('')
    result = []
    map = {}
    for word1 in text:
        sentence = word1.split()
        for word2 in sentence:
            word2 = word2.translate(str.maketrans('','',string.punctuation))
            result.append(word2.lower())
    result = [w for w in result if w not in stop_words]
    for word in result:
        if word in map:
            map[word] += 1
        else:
            map[word] = 1
    map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))
    print(map)
    ws = pd.Series(map, name='Count')
    ws.index.name = 'Word'
    ws.reset_index()
    ws20 = ws.head(20)
    fig = px.bar(ws20)
    fig.show()

def get_wordcount_mod(text):
    stop_words = list(stopwords.words('english'))
    stop_words.append('')
    result = []
    final_words = []
    for sentence in text:
        #print(sentence)
        #print('-'*80)
        #sentence = ''.join(sentence)
        #sentence = sentence.lower()
        #sentence = word_tokenize(sentence)
        #sentence = [w for w in sentence if w not in stop_words]
        vectorizer = TfidfVectorizer(use_idf=True)
        try:
            X = vectorizer.fit_transform(sentence)
        except ValueError:
            continue
        word_map = {}
        for word, score in zip(vectorizer.get_feature_names_out(), X.toarray()[0]):
            if score > 0.0:
                word_map[word] = score
        sort_words = sorted(word_map.items(), key=lambda x: x[1], reverse=True)
        result.append([word[0] for word in sort_words])
    #print(result)
    result_words = [w for ws in result for w in ws]
    print(result_words)
    create_bar(result_words)


def create_bar(text_list):
    stop_words = list(stopwords.words('english'))
    stop_words.append('')
    text_list = [w for w in text_list if w not in stop_words]
    map = {}
    for word in text_list:
        if word in map:
            map[word] += 1
        else:
            map[word] = 1
    map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))
    print(map)
    ws = pd.Series(map, name='Count')
    ws.index.name = 'Word'
    ws.reset_index()
    ws20 = ws.head(30)
    fig = px.bar(ws20)
    fig.show()

def get_sentiment(sentence):
    text_map = {'Text': []}
    for text in (sentence):
        if text != '':
            text_map['Text'].append(text)
    data_text = pd.DataFrame(text_map)
    preprocessor = Full_process()
    slang = preprocessor.slang_load('SlangLookupTable.txt')
    data_text['Text'] = data_text['Text'].apply(lambda x: preprocessor.slang_switch(x, slang))
    data_text['Text'] = data_text['Text'].apply(preprocessor.remove_unneeded)
    data_text['Text'] = data_text['Text'].apply(preprocessor.preprocess)
    #data_text['Text'] = data_text['Text'].apply(preprocessor.add_negation)
    data_text[['scores_neg', 'scores_neu', 'scores_pos', 'scores_compound']] = data_text['Text'].apply(lambda x: pd.Series(nltk_sentiment(x)))
    #data_text['scores2'] = data_text['Text'].apply(spacy_sentiment)
    display(data_text)
    #display(data_text['scores1'])


    print(f"Negative: {data_text['scores_neg'].mean()}")
    print(f"Neutral: {data_text['scores_neu'].mean()}")
    print(f"Positive: {data_text['scores_pos'].mean()}")
    print(f"Compound(No zeros): {data_text.loc[data_text['scores_compound'] != 0.0, 'scores_compound'].mean()}")
    print(f"Compound: {data_text['scores_compound'].mean()}")
    data_text.to_csv('data.csv')



def create_wordplot(text):
    wc = WordCloud(width = 1200, height = 600, background_color='black', max_font_size = 200)
    wc.generate(''.join(text))
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


#sub = input('Subreddit (Type Nothing for general search)')
#search = input('Search Term:')

results_post, results_text = get_sub_info('DeSantis', '')
get_wordcount_mod(results_post)
get_sentiment(results_text)
create_wordplot(results_text)