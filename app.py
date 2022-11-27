import streamlit as st
st.set_page_config(page_title='ADHD', page_icon="🤯", layout="wide")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set
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
st.write("ADHD \n Something about ADHD")

#read in dataset
df = pd.read_csv('datasets/KKI_phenotypic.csv')

#dropping unrelevant columns
df = df.drop(columns = ['Site', 'ADHD Measure', 'IQ Measure', 'Full2 IQ', 'QC_Rest_1', 'QC_Rest_2', 'QC_Rest_3', 'QC_Rest_4', 'QC_Anatomical_1', 'QC_Anatomical_2'])

#Round age for fewer unique values and making into integer
df['Age'] = df['Age'].round(decimals = 0)
df['Age'] = df['Age'].astype(int)

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


#for the sliders we are using a new value called df_age
df_age = df
Age_selected = st.slider("Select Age", min_value = int(df_age.Age.min()), max_value= int(df_age.Age.max()), value = (0,100), step=1)
df_age = df_age[(df_age.Age > Age_selected[0]) & (df_age.Age < Age_selected[1])]

#filter for country - set to a sidebar
st.sidebar.title("Gender ♂️♀️")
gender_select = st.sidebar.multiselect("What gender do you want? (0 is male, 1 is female)",(0, 1))

df_age = df_age[df_age['Gender'].isin(gender_select)]

#line chart for marriage
c = alt.Chart(df_age).mark_circle().encode(
    
    alt.X('Age:N',axis=alt.Axis(values=["Age_selected"]),
        scale=alt.Scale(zero=False),
    ),
    y='ADHD Index',
    color=alt.value("Gender")
)
st.altair_chart(c, use_container_width=True)


#from 1900 to 2018
adhd_gender = df.groupby('Gender')['Gender'].count()
adhd_gender = adhd_gender[(adhd_gender.index == 1) | (adhd_gender.index==0)]

gender_fig = px.pie(adhd_gender, values=adhd_gender.values, names=adhd_gender.index)
gender_fig.update_layout(height=500, width=600)
gender_fig.show()