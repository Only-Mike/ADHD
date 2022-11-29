import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title='ADHD', page_icon="🤯", layout="centered")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set
import plotly.express as px
import altair as alt
from tqdm import tqdm
from stqdm.stqdm import stqdm
import preprocessor as prepro
import itertools

import spacy #spacy for quick language prepro
nlp = spacy.load('en_core_web_sm') #instantiating English module

# sampling, splitting
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


# loading ML libraries
from sklearn.pipeline import make_pipeline #pipeline creation
from sklearn.feature_extraction.text import TfidfVectorizer #transforms text to sparse matrix
from sklearn.linear_model import LogisticRegression #Logit model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report #that's self explanatory
from sklearn.decomposition import TruncatedSVD #dimensionality reduction
from sklearn.preprocessing import StandardScaler #Scaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import umap 


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


#Title and subheader
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

@st.experimental_singleton
def read_process_data():
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
    synthetic_data = model.sample(2000)

    #Combining the two datasets
    df = pd.concat([synthetic_data, df])
    return df

df = read_process_data()

#"""Visualizing the app"""
st.subheader("In this section you can play around with the data avaible to us")
st.write("Note: As the dataset we have used didn't have many entries and was limited to age, we have used something called SDV to synthetise some more data. This means it has created data from itself and therefore better visualtions. BUT this also means the data isn't 100% true to real life until more data is avaible")
st.write("If you have ADHD or ADD you can help us get more data by answering these questions")
st.caption("Link will come soon", unsafe_allow_html=True)

#Tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Age and ADHD", "Predictor - Do you have ADHD?", "Topic Modeling", "ADHD Clusters"])

with tab1:
    st.header("Introduction to this app")
    st.subheader("More is coming soon")

with tab2:
    st.header("Age and ADHD")
    st.subheader("Still WIP")

    #sliders
    Age_selected = st.slider("Select Age", min_value = int(df.Age.min()), max_value= int(df.Age.max()), value = (0,100), step=1)
    df = df[(df.Age > Age_selected[0]) & (df.Age < Age_selected[1])]

    #filter for country - set to a sidebar
    st.sidebar.title("Gender ♂️♀️")
    gender_select = st.sidebar.multiselect("Select Gender ♂️♀️", ("Female", "Male"))

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

with tab3:
    st.title("Predictor - Do you have ADHD?")
    st.subheader("Try out our ADHD predictor!")
    st.write("NB: This is not a real medicaltest, just a test")
    df = read_process_data()

    #Define X and y
    X = df[["Inattentive", "Hyper/Impulsive"]].values
    y = df["ADHD Index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    model = LinearRegression()

    model.fit(X_train, y_train)

    in_lvl = st.slider(label="How would you rank your inattentive level from 1 to 10?", min_value=1, max_value=10)
    hy_lvl = st.slider(label="How would you rank your hyper level/impulsive from 1 to 10?", min_value=1, max_value=10)
    #Imp_lvl = st.slider(label="How would you rank your impulsive level from 1 to 10?", min_value=1, max_value=10)


    if st.button('Predict'):
        #imp_hy_lvl = (hy_lvl+Imp_lvl)/2
        X_new = [[in_lvl, hy_lvl]]
        adhd_output_index = model.predict(X_new) *10
        st.write( "ADHD index is", adhd_output_index)


with tab4:
    st.title("Topic Modeling")
    st.subheader("On this page you'll see the most popular topics for the last 5 years")
    
    data1 = pd.read_csv('datasets/adhd2018.csv')
    data2 = pd.read_csv('datasets/adhd2019.csv')
    data3 = pd.read_csv('datasets/adhd2020.csv')
    data4 = pd.read_csv('datasets/adhd2021.csv')
    data5 = pd.read_csv('datasets/adhd2022.csv')
    frames = [data1, data2, data3, data4, data5] #creating frame for all datasets
    data = pd.concat(frames) #Concat all datasets to "df"

    # prepro settings
    prepro.set_options(prepro.OPT.URL, prepro.OPT.NUMBER, prepro.OPT.RESERVED, prepro.OPT.MENTION, prepro.OPT.SMILEY)

    data = data[['Authors', 'Author(s) ID','Title', 'Abstract','Year', 'Cited by']]
    
    #Take a random sample of 2500 papers. This is for making the model run faster.
    data = data.sample(n=2500)

    data['text'] = data['Abstract']
    #Cleaning the text
    data['text_clean'] = data['text'].map(lambda t: prepro.clean(t))


    clean_text = []

    pbar = tqdm.tqdm(total=len(data['text_clean']),position=0, leave=True)

    for text in nlp.pipe(data['text_clean'], disable=["tagger", "parser", "ner"]):

        txt = [token.lemma_.lower() for token in text 
            if token.is_alpha 
            and not token.is_stop 
            and not token.is_punct]

        clean_text.append(" ".join(txt))

        pbar.update(1)

    # write everything into a single function for simplicity later on
    def text_prepro(texts):
        texts_clean = texts.map(lambda t: prepro.clean(t))
        clean_container = []
        pbar = tqdm.tqdm(total=len(texts_clean),position=0, leave=True)
        for text in nlp.pipe(texts_clean, disable=["tagger", "parser", "ner"]):
            txt = [token.lemma_.lower() for token in text 
                if token.is_alpha 
                and not token.is_stop
                and not token.is_punct]
            clean_container.append(" ".join(txt))
            pbar.update(1)
        return clean_container
    
    data['text_clean'] = text_prepro(data['text'])

    tokens = []
    for summary in nlp.pipe(data['text_clean'], disable=["ner"]):
        proj_tok = [token.lemma_.lower() for token in summary
                    if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']
                    and not token.is_stop
                    and not token.is_punct]
        tokens.append(proj_tok)

    data['tokens'] = tokens

    # Create a Dictionary from the articles: dictionary
    dictionary = Dictionary(data['tokens'])

    # filter out low-frequency / high-frequency stuff, also limit the vocabulary to max 1000 words
    dictionary.filter_extremes(no_below=2, no_above=0.2, keep_n=1000)

    # construct corpus using this dictionary
    corpus = [dictionary.doc2bow(doc) for doc in data['tokens']]
    lda_model = LdaMulticore(corpus, id2word=dictionary, num_topics=12, workers = 4, passes=10)
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(lda_display)


with tab5:
    st.title("ADHD Clustering")
    st.subheader("This app is made by Snorre and Mike")
    st.write("ADHD \n Something about ADHD clustering")
    df = read_process_data()


    #We will scale the data which requires the following tool
    scaler = StandardScaler()

    df = df.loc[~df.index.duplicated(), :]

    # with the scaler.fit_transfor we learn x-y relationships and transform the data.
    df_scaled = scaler.fit_transform(df)

    #Age pre-scaling
    sns.displot(data=df, x="Age", kind="kde")

    #Age post-scaling
    sns.displot(data=pd.DataFrame(df_scaled, columns=df.columns), 
                x="Age",
                kind="kde")

    #initialize PCA
    pca = PCA(n_components=2)

    # fit and transform the data
    df_reduced_pca = pca.fit_transform(df_scaled)

    #Reduced data chart
    vis_data = pd.DataFrame(df_reduced_pca)
    vis_data['Age'] = df['Age']
    vis_data['ADHD Index'] = df['ADHD Index']
    vis_data.columns = ['x', 'y', 'Age', 'ADHD Index']

    chart_data = pd.DataFrame(vis_data)

    c = alt.Chart(chart_data).mark_circle().encode(
        x='Age', y='ADHD Index', size='ADHD Index', color='Age', tooltip=['Age', 'ADHD Index'])

    st.altair_chart(c, use_container_width=True)

    st.subheader("Correlation heatmap")
    st.write("This shows the correlation between different values in the dataset")

    #Correlation heatmap
    fig1 = plt.figure(figsize=(18,2))
    sns.heatmap(pd.DataFrame(pca.components_, columns=df.columns), annot=True)
    st.pyplot(fig1)



    st.subheader("UMAP and K-means clustering")
    st.write("For this visualization, we have used K-means and UMAP")

    with st.spinner('Proccesing data and creating cluster...'):
        umap_scaler = umap.UMAP()
        embeddings = umap_scaler.fit_transform(df_scaled)

        #Clearly there is some difference between people with a secondary dianosis and those without
        #fig2 = rcParams['figure.figsize'] = 15,10
        #sns.scatterplot(embeddings[:,0],embeddings[:,1], color = df['Secondary Dx '])
        #st.pyplot(fig2)


        #K-means clustering
        from sklearn.cluster import KMeans

        #def cluster_umap_kmeans

        clusterer = KMeans(n_clusters=3)

        Sum_of_squared_distances = []
        K = range(1,10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df_scaled)
            Sum_of_squared_distances.append(km.inertia_)

        #Umap scaler 
        umap_scaler_km = umap.UMAP(n_components=3)
        embeddings_km = umap_scaler.fit_transform(df_scaled)


        Sum_of_squared_distances = []
        K = range(1,10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(embeddings_km)
            Sum_of_squared_distances.append(km.inertia_)


        clusterer.fit(df_scaled)
        df['cluster'] = clusterer.labels_
        df.groupby('cluster').Inattentive.mean()


        vis_data1 = pd.DataFrame(embeddings)
        vis_data1['Gender'] = df['Gender']
        vis_data1['cluster'] = df['cluster']
        vis_data1['Secondary Dx '] = df['Secondary Dx ']
        vis_data1.columns = ['x', 'y', 'Gender', 'cluster','Secondary Dx ']



        c1 = alt.Chart(vis_data1).mark_circle(size=60).encode(
            x='x',
            y='y',
            tooltip=['Gender', 'Secondary Dx '],
            color=alt.Color('cluster:N', scale=alt.Scale(scheme='dark2'))
        ).interactive()

        st.altair_chart(c1, use_container_width=True)


        #wrap up 
    st.success('Done!')