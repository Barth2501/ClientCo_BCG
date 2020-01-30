# import des packages utilisés

import argparse
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from clustering import *
from churn_detection import * 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help="Import the dataset you want to use", type=str)
parser.add_argument('--mode', '-m',help="Prediction or Train", type=str)
parser.add_argument('--cluster', '-c',help="Cluster you want to predict", type=int)
args = parser.parse_args()

global df

if __name__ == "__main__":
    df, cluster_client_df = load_df(args.dataset)
    final_label_df = get_clustering(cluster_client_df)
    final_label_df = final_label_df.drop(['kmeans_label','index'],axis=1).rename({'second_kmeans_label':'cluster'},axis=1)
    df = df.merge(final_label_df, on='client_id')

    df = df.loc[df.cluster == args.cluster]
    print('---Labellisation du dataframe---')
    df = labelize(args.cluster,df)

    print('---Modèle de churn---')
    util_df = build_dataset(df)
    if args.mode == 'train':
        churn_train(util_df,args.cluster)
    elif args.mode == 'predict':
        # Retourne un dataframe de client triés selon la probabilité de churner
        churner = predict(util_df,args.cluster)
        print(churner)