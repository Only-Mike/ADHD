import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import umap 
from sklearn.preprocessing import StandardScaler

st.title("ADHD Clustering")
st.subheader("This app is made by Snorre and Mike")
st.write("ADHD \n Something about ADHD clustering")

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
df = df.pd.concat([synthetic_data])

#We will scale the data which requires the following tool
scaler = StandardScaler()

df = df.loc[~df.index.duplicated(), :]

# with the scaler.fit_transfor we learn x-y relationships and transform the data.
df_scaled = scaler.fit_transform(df)

#Age pre-scaling
sns.displot(data=df, 
            x="Age", 
            kind="kde")

#Age post-scaling
sns.displot(data=pd.DataFrame(df_scaled, columns=df.columns), 
            x="Age",
            kind="kde")

#import at initialize PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# fit and transform the data
df_reduced_pca = pca.fit_transform(df_scaled)

import altair as alt
vis_data = pd.DataFrame(df_reduced_pca)
vis_data['Gender'] = df['Gender']
vis_data['Age'] = df['Age']
vis_data.columns = ['x', 'y', 'Gender', 'Age']

#Correlation heatmap
plt.figure(figsize=(18,2))
sns.heatmap(pd.DataFrame(pca.components_, columns=df.columns), annot=True)