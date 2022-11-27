import streamlit as st
st.set_page_config(page_title='ADHD', page_icon="🤯", layout="centered")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import seaborn as sns; sns.set
import plotly.express as px
from tqdm import tqdm
import preprocessor as prepro # text prepro

import spacy #spacy for quick language prepro
nlp = spacy.load('en_core_web_sm') #instantiating English module

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

st.title("ADHD🧠")
st.subheader("This app is made by Snorre and Mike")
st.write("Before diving into this app, we highly recommend (if you don't know what it is already) diving into what ADHD is")
st.write("If you don't know what ADHD is about we have linked some 2 minute videos going through the basics of ADHD and what symptoms one might have")

def shorten_vid_option(opt):
    return opt.split("/")[-1]

vidurl = st.selectbox(
    "Pick a video to play",
    (
        "https://www.youtube.com/watch?v=9TcNQkyxMj8&ab_channel=AmericanPsychiatricAssociation",
        "https://www.youtube.com/watch?v=w8JnDhp83gA&ab_channel=NeuroscientificallyChallenged"
    ),
     0,
    shorten_vid_option,
)
st.video(vidurl, format="video/mp4", start_time=0 )


#"""In this section we clean the data"""
#read in dataset
df = pd.read_csv('datasets/KKI_phenotypic.csv')

#dropping unrelevant columns
df = df.drop(columns = ['Site', 'ADHD Measure', 'IQ Measure', 'Full2 IQ', 'QC_Rest_1', 'QC_Rest_2', 'QC_Rest_3', 'QC_Rest_4', 'QC_Anatomical_1', 'QC_Anatomical_2'])

#Round age for fewer unique values and making into integer
df['Age'] = df['Age'].round(decimals = 0)
df['Age'] = df['Age'].astype(int)

#Making gender from 0 and 1 to Female and Male
df['Gender'].replace(('1', '0'), ("Male", "Female"), inplace=True)

#Removes rows with -999 in the following columns
df = df[df['Inattentive'] != -999]
df = df[df['Hyper/Impulsive'] != -999]
df = df[df['ADHD Index'] != -999]

#Making none secondary dx into 0 and any secondary dx into 1
df['Secondary Dx '].replace(('Simple phobia', 'Simple Phobia', 'simple phobias', 'ODD', 'Simple Phobia ', 'ODD; Phobia', 'Specific phobia', 'Phobia', 'social and simple phobia '), (1, 1, 1, 1, 1, 1, 1, 1, 1), inplace=True)
df['Secondary Dx '] = df['Secondary Dx '].fillna(0).astype(int)

#import Synthetic data creator SDV
from sdv.tabular import GaussianCopula
model = GaussianCopula()
model.fit(df)

#Creating the synthetic data
synthetic_data = model.sample(500)
synthetic_data.head()

#Combining the two datasets
df = pd.concat([synthetic_data, df])




#"""Visualizing the app"""
st.subheader("In this section you can play around with the data avaible to us")
st.write("Note: As the dataset we have used didn't have many entries and was limited to age, we have used something called SDV to synthetise some more data. This means it has created data from itself and therefore better visualtions. BUT this also means the data isn't 100% true to real life until more data is avaible")
st.write("You can help us get more data by answering these questions")
st.caption("Link will come soon", unsafe_allow_html=True)


#sliders
Age_selected = st.slider("Select Age", min_value = int(df.Age.min()), max_value= int(df.Age.max()), value = (0,100), step=1)
df = df[(df.Age > Age_selected[0]) & (df.Age < Age_selected[1])]

#filter for country - set to a sidebar
st.sidebar.title("Gender ♂️♀️")
gender_select = st.sidebar.multiselect("What gender do you want?",("Female", "Male"))

df = df[df.Gender.isin(gender_select)]

#line chart for age vs gender
c = alt.Chart(df).mark_circle().encode(
    
    alt.X('Age:N',axis=alt.Axis(values=["Age_selected"]),
        scale=alt.Scale(zero=False),
    ),
    y='ADHD Index',
    color=alt.value("Gender"),
    tooltip=['Age:N', 'Gender', 'ADHD Index'])

st.altair_chart(c, use_container_width=True)