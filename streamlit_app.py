import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic import BERTopic
model = BERTopic(verbose=True)
docs = [str(text) for text in df['Text'].tolist()] 
topics, probabilities = model.fit_transform(docs)
model.get_topic_freq().head(11)
model.get_topic(6)
model.visualize_topics()
