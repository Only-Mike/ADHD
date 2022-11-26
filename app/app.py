import streamlit as st
import numpy as np
import pandas as pd
#from tqdm import tqdm
#import preprocessor as prepro # text prepro

#import spacy #spacy for quick language prepro
#nlp = spacy.load('en_core_web_sm') #instantiating English module

# sampling, splitting
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


# loading ML libraries
from sklearn.pipeline import make_pipeline #pipeline creation
from sklearn.feature_extraction.text import TfidfVectorizer #transforms text to sparse matrix
from sklearn.linear_model import LogisticRegression #Logit model
from sklearn.metrics import classification_report #that's self explanatory
from sklearn.decomposition import TruncatedSVD #dimensionality reduction
from xgboost import XGBClassifier

import altair as alt #viz

#explainability
import eli5
from eli5.lime import TextExplainer

# topic modeling

from gensim.corpora.dictionary import Dictionary # Import the dictionary builder
from gensim.models import LdaMulticore # we'll use the faster multicore version of LDA

#Import pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


st.set_page_config(page_title='ADHD', page_icon="shocked_face_with_exploding_head", layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("ADHD")







if st.button('Generate pyLDAvis'):
            with st.spinner('Creating pyLDAvis Visualization ...'):
                py_lda_vis_data = pyLDAvis.gensim_models.prepare(st.session_state.model, st.session_state.corpus,
                                                                 st.session_state.id2word)
                py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
            with st.expander('pyLDAvis', expanded=True):
                st.markdown('pyLDAvis is designed to help users interpret the topics in a topic model that has been '
                            'fit to a corpus of text data. The package extracts information from a fitted LDA topic '
                            'model to inform an interactive web-based visualization.')
                st.markdown('https://github.com/bmabey/pyLDAvis')
                components.html(py_lda_vis_html, width=1300, height=800)
