"""
This file contains code from https://github.com/lukasruff/CVDD-PyTorch/blob/aa2b033ed8216ce132ef6977da1e4fae665fb0c0/src/utils/misc.py#L94
The repo contains code from the following work:
Ruff et. al - Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text
ACL '19
"""
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text