import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
df = pd.concat([synthetic_data, df])

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

#Reduced data chart
import altair as alt
vis_data = pd.DataFrame(df_reduced_pca)
vis_data['Age'] = df['Age']
vis_data['ADHD Index'] = df['ADHD Index']
vis_data.columns = ['x', 'y', 'Age', 'ADHD Index']

chart_data = pd.DataFrame(vis_data)

c = alt.Chart(chart_data).mark_circle().encode(
    x='Age', y='ADHD Index', size='ADHD Index', color='Age', tooltip=['Age', 'ADHD Index'])

st.altair_chart(c, use_container_width=True)


#Correlation heatmap
fig1 = plt.figure(figsize=(18,2))
sns.heatmap(pd.DataFrame(pca.components_, columns=df.columns), annot=True)
st.pyplot(fig1)



umap_scaler = umap.UMAP()
embeddings = umap_scaler.fit_transform(df_scaled)

#Clearly there is some difference between people with a secondary dianosis and those without
#fig2 = rcParams['figure.figsize'] = 15,10
#sns.scatterplot(embeddings[:,0],embeddings[:,1], color = df['Secondary Dx '])
#st.pyplot(fig2)


#K-means clustering
from sklearn.cluster import KMeans
clusterer = KMeans(n_clusters=6)

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_scaled)
    Sum_of_squared_distances.append(km.inertia_)

#Umap scaler 
umap_scaler_km = umap.UMAP(n_components=6)
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