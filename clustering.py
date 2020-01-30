import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import joblib

global df

def load_df(df_path):
    # import du dataset
    print('--import du dataset--')
    df = pd.read_csv(df_path,sep=';' ,parse_dates=['date_order','date_invoice'], dtype= {'product_id':'uint32','client_id':'uint32','sales_net':'float32','branch_id':'uint16','quantity':'int32'})
    print('---Encodage de la colonne channel---')
    le = LabelEncoder()
    df.order_channel = le.fit_transform(df.order_channel)
    print('---Retraitement du dataset---')
    df['delay'] = df.date_invoice - df.date_order
    # drop the row with na
    df = df.dropna()
    df = df.assign(delay=df.delay.dt.days)
    df.delay = df.delay.astype('int16')
    # drop unused columns
    df = df.drop('Unnamed: 0',axis=1)
    df = df.drop('date_invoice',axis=1)

    print('---Construction des features---')
    ## Construction des features 
    # quantité de produits différents par client
    quantity_unique_df = df[['client_id','product_id']].groupby(['client_id']).agg({'product_id':pd.Series.nunique})
    cluster_client_df = pd.DataFrame(quantity_unique_df).rename({'product_id':'nb_of_unique_product'},axis=1)
    # Depuis combien de temps un client n'a pas commandé
    since_last_order_df = df[['date_order','client_id']].groupby('client_id').date_order.max()
    since_last_order_df = df.date_order.max() - since_last_order_df
    since_last_order_df = since_last_order_df.map(lambda x: x.days)
    cluster_client_df = cluster_client_df.join(since_last_order_df).rename({'date_order':'time_since_last_order'},axis=1)
    # revenu total par client
    total_revenue_df = df[['client_id','sales_net']].groupby('client_id').sum()
    cluster_client_df = cluster_client_df.join(total_revenue_df).rename({'sales_net':'total_revenue'},axis=1)
    # revenu moyen par commande par client
    revenue_per_command_df = df[['date_order','client_id','sales_net']].groupby(['client_id','date_order']).agg({'sales_net':'sum'}).groupby('client_id').agg({'sales_net':'mean'})
    cluster_client_df = cluster_client_df.join(revenue_per_command_df).rename({'sales_net':'revenue_per_command'},axis=1)
    # quantité moyenne par commande pour chaque client
    quantity_per_command_df = df[['date_order','client_id','quantity']].groupby(['client_id','date_order']).agg({'quantity':'sum'}).groupby('client_id').agg({'quantity':'mean'})
    cluster_client_df = cluster_client_df.join(quantity_per_command_df).rename({'quantity':'quantity_per_command'},axis=1)
    # Fréquence d'achat par client
    nb_order_df = df[['client_id','date_order']].groupby('client_id').agg({'date_order':pd.Series.nunique}).rename({'date_order':'nb_order'},axis=1)
    client_window_df = df[['date_order','client_id']].groupby('client_id').date_order.max()-df[['date_order','client_id']].groupby('client_id').date_order.min()
    client_window_df = client_window_df.map(lambda x: x.days)
    order_freqency_df = nb_order_df.nb_order/client_window_df
    order_freqency_df = order_freqency_df.rename('order_frequency')
    order_freqency_df.loc[order_freqency_df==np.inf] = 0
    cluster_client_df = cluster_client_df.join(order_freqency_df)
    # Nombre de canaux utilisés pour chaque client
    client_nb_channel_df = df[['client_id','order_channel']].groupby('client_id').agg({'order_channel':pd.Series.nunique})
    cluster_client_df = cluster_client_df.join(client_nb_channel_df).rename({'order_channel':'nb_channel_used'},axis=1)
    # Canal favori par client en one hot encoding (get_dummies) et nombre de fois qu'il l'a utilisé
    client_favorite_channel = df[['client_id','order_channel','date_order']].groupby(['client_id','order_channel']).agg({'date_order':pd.Series.nunique})
    client_favorite_channel_df = client_favorite_channel.groupby('client_id').agg({'date_order':'max'}).reset_index().merge(client_favorite_channel.reset_index(), on=['client_id','date_order']).drop_duplicates(['client_id','date_order'])
    client_favorite_channel_df = client_favorite_channel_df.rename({'date_order':'time_fav_channel_used','order_channel':'fav_channel'},axis=1).set_index('client_id')
    # Encodage des données et retrait du canal other pour eviter la redondance
    split_columns = pd.get_dummies(client_favorite_channel_df.fav_channel).drop(3,axis=1)
    cluster_client_df = cluster_client_df.join(client_favorite_channel_df.time_fav_channel_used)
    cluster_client_df = cluster_client_df.join(split_columns)
    return df, cluster_client_df

def cluster_scaling(cluster_client_df):
    print('---Scaling---')
    # Scaling du dataset
    scaler = MinMaxScaler()
    X = cluster_client_df.values
    X = scaler.fit_transform(X)
    return X

def train_first_clustering(X):
    print('---1er Kmeans---')
    # Kmeans : 1er Algo clustering non hierarchique au vue de réduire les clusters
    kmeans = KMeans(n_clusters=500, random_state=0).fit(X)
    filename = 'modele/first_clustering_model.sav'
    joblib.dump(kmeans, filename)
    return kmeans

def train_second_clustering(second_X):
    print('---2e Kmeans---')
    # 2eme Kmeans avec 3 clusters
    second_kmeans = KMeans(n_clusters=3, random_state=0).fit(second_X)
    filename = 'modele/second_clustering_model.sav'
    joblib.dump(second_kmeans, filename)
    return second_kmeans

def get_clustering(cluster_client_df):
    X = cluster_scaling(cluster_client_df)
    # Load first clustering model
    filename = 'modele/first_clustering_model.sav'
    try:
        print('---Récuération du 1er modèle kmeans---')
        kmeans = joblib.load(filename)
    except:
        print('---Construction du 1er modèle kmeans---')
        kmeans = train_first_clustering(X)
    labels_df = pd.DataFrame(kmeans.predict(X),columns = ['kmeans_label'])
    labels_df.index = cluster_client_df.index

    # Dataset post-clustering non hierarchique des centres des clusters (500)
    cluster_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(cluster_client_df.columns))
    second_X = cluster_df.values
    filename = 'modele/second_clustering_model.sav'
    try:
        print('---Récupération du 2e modèle kmeans---')
        second_kmeans = joblib.load(filename)
    except:
        print('--Construction du 2e modèle kmeans---')
        second_kmeans = train_second_clustering(second_X)
    second_labels_df = pd.DataFrame(second_kmeans.predict(second_X),columns = ['second_kmeans_label']).reset_index()

    # Obtention des clusters de client par left join des résultats du 2e clustering non hierarchique
    final_label_df = labels_df.reset_index().merge(second_labels_df, left_on='kmeans_label', right_on='index', how='left')
    return final_label_df

# fonction permettant de calculer les distances entre chaque cluster en vue d'afficher le dendogramme
def get_distances(X,model,mode='l2'):
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5 
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d


        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append( wNew)
    return distances, weights

def show_dendogram(cluster_client_df):
    X = cluster_scaling(cluster_client_df)
    # Load first clustering model
    filename = 'modele/first_clustering_model.sav'
    try:
        kmeans = joblib.load(filename)
    except:
        kmeans = train_first_clustering(X)
    labels_df = pd.DataFrame(kmeans.predict(X),columns = ['kmeans_label'])
    labels_df.index = cluster_client_df.index
    # Dataset post-clustering non hierarchique des centres des clusters (500)
    cluster_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(cluster_client_df.columns))
    second_X = cluster_df.values
    # On utilise ici le critère de Ward car issue d'aggrégation de clusters
    model = AgglomerativeClustering(n_clusters=2,linkage="ward")
    model.fit(second_X)
    distance, weight = get_distances(second_X,model)
    linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
    plt.figure(figsize=(20,10))
    dendrogram(linkage_matrix)
    plt.show()

def ACP(cluster_client_df):
    X = cluster_scaling(cluster_client_df)
    # Load first clustering model
    filename = 'modele/first_clustering_model.sav'
    try:
        kmeans = joblib.load(filename)
    except:
        kmeans = train_first_clustering(X)
    labels_df = pd.DataFrame(kmeans.predict(X),columns = ['kmeans_label'])
    labels_df.index = cluster_client_df.index
    # Dataset post-clustering non hierarchique des centres des clusters (500)
    cluster_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(cluster_client_df.columns))
    second_X = cluster_df.values
    filename = 'modele/second_clustering_model.sav'
    try:
        second_kmeans = joblib.load(filename)
    except:
        second_kmeans = train_second_clustering(second_X)
    print('---Analyse en composantes principales---')
    # affichage des clusters dans l'espace des composantes principales
    pca = PCA(n_components=6)
    pca_X = pca.fit_transform(second_X)

    print('---Affichage des clusters en 3 dimensions---')
    # affichage selon les 3 axes principaux
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_X[:,0],pca_X[:,1],pca_X[:,2], c=second_kmeans.labels_)
    plt.show()
    print(pd.DataFrame(pca.components_,columns=cluster_client_df.columns))