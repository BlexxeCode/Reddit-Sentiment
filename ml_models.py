import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nltk.download('vader_lexicon')


def nltk_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

def spacy_sentiment(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    doc = nlp(text)
    scores = doc._.blob.polarity

    return scores

def nltk_spacy(text):
    pass

def trained_sentinemt(text):
    pass



#sen = "I don't like this at all!"
#tr = nltk_sentiment(sen)
#print(tr)