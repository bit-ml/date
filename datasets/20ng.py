from sklearn.datasets import fetch_20newsgroups
import os
import re
import nltk
import string
from clean_text import clean_text

def export_ds(subset, groups):
    for entry in groups:
        dataset, _ = fetch_20newsgroups(
            data_home='./20ng_od', subset=subset, categories=entry['names'],
            remove=('headers', 'footers', 'quotes'), return_X_y=True)
        corpus = ''
        for article in dataset:
            stripped = re.sub('\s+',' ', article)
            stripped = clean_text(stripped)
            corpus += f'\n\n{stripped}'

        full_path = os.path.join('./20ng_od', subset)

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        with open(os.path.join(full_path,f'{entry["topic"]}.txt'), 'w') as f:
            f.write(corpus)

groups = [
    {
        'names': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'comp.windows.x'],
        'topic': 'comp'
    },
    {
        'names': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
        'topic': 'rec'
    },
    {   'names': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
        'topic': 'sci'
    },
    {
        'names': ['misc.forsale'],
        'topic': 'misc'
    },
    {
        'names': ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
        'topic': 'pol'
    },
    {
        'names': ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'],
        'topic': 'rel'
    }
]
export_ds('train', groups)
export_ds('test', groups)
