import pandas as pd
import regex
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


EMAIL_REGEX_STR = '\S*@\S*'
MENTION_REGEX_STR = '@\S*'
HASHTAG_REGEX_STR = '#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', text) for text in texts]
    docs = [[w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('english')] for doc in texts]
    return docs
import pandas as pd
import spacy


def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)

    # bigram / trigam preprocessing ...

    lemmantized_docs = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for doc in docs:
        doc = nlp(' '.join(doc))
        lemmantized_docs.append([token.lemma_ for token in doc])
import gensim
from gensim import corpora


def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus 
import gensim
from gensim import corpora


def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


def train_model(docs, num_topics: int = 10, per_word_topics: bool = True):
    id2word, corpus = prepare_training_data(docs)
    model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, per_word_topics=per_word_topics)
    return model


def train_model(docs, num_topics: int = 10, per_word_topics: bool = True):
    id2word, corpus = prepare_training_data(docs)
    model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, per_word_topics=per_word_topics)
    return model
