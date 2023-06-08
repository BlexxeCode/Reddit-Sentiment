import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sacremoses import MosesDetokenizer


class Full_process():
    def remove_unneeded(self, text):
        pattern = re.compile(r'http[s]?://\S+')
        return re.sub(pattern, '', text)

    def preprocess(self, text):
        text = text.lower()

        stop_words = stop_words = list(stopwords.words('english'))
        stop_words.append('')
        words = text.split()
        word = [w for w in words if w not in stop_words]
        text = ' '.join(words)

        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words]
        text = ' '.join(words)

        return text


    def slang_load(self,file):
        slang_dict = {}
        with open(file, 'r') as file:
            for line in file:
                slang, meaning = line.strip().split('-')
                if slang.lower() not in slang_dict:
                    slang_dict[slang.lower()] = meaning.lower()
        return slang_dict

    def slang_switch(self, text, slang_load):
        tokens = word_tokenize(text)
        for i, token in enumerate(tokens):
            if token.lower() in slang_load:
                tokens[i] = slang_load[token.lower()]
        detokenizer = MosesDetokenizer('en')
        text = detokenizer.detokenize(tokens)

        return text

    def add_negation(self,text):
        negation_words = ["not", "n't", "no", "never", "nothing", "nowhere", "none", "neither", "nor"]

        tokens = word_tokenize(text)
        new_tokens = []
        negation = False
        for i in range(len(tokens)):
            if tokens[i] in negation_words:
                negation = True
            elif negation and tokens[i] not in [',', '.', '?', '!', ';', ':', '-', ')', ']', '}']:
                new_tokens.append('not_' + tokens[i])
            else:
                new_tokens.append(tokens[i])
                negation = False
        return ' '.join(new_tokens)



#dit = Full_process()
#d = dit.slang_load('SlangLookupTable.txt')
#r = dit.slang_switch('lol, that is hilarious, lmao',d)
#print(r)