import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import joblib

def build_dataset(df):
    print('---Aggrégation par client---')
    # aggregation par client
    client_df = df.groupby(['client_id']).agg({'label':'max','delay':'mean','quantity':'mean','sales_net':'mean','date_order':'max'})
    client_df = client_df.rename({'delay':'delay_mean','quantity':'mean_quantity','sales_net':'mean_sales_net','date_order':'last_order_date'},axis=1)

    print('---Récupération des tendances par client---')
    # ajout des trend last, last 2 and last 5
    test = df.groupby(['client_id','date_order']).agg({'delay':'mean','sales_net':'mean','quantity':'mean','branch_id':pd.Series.nunique,'product_id':pd.Series.nunique}).reset_index()
    # rajout des variables indiquant le nombre moyen de canaux et produits differents utilisés par commande 
    client_df = client_df.join(test.reset_index().groupby('client_id').agg({'branch_id':'mean','product_id':'mean'}))
    client_df = client_df.rename({'branch_id':'mean_nb_of_branch','product_id':'mean_nb_of_product'},axis=1)
    # rajout des variables d'evolution de la derniere commande
    last_order_df = test.rolling(2).agg({'delay':'mean','sales_net':'mean','quantity':'mean','branch_id':'mean','product_id':'mean'})
    test = test.join(last_order_df,rsuffix='_last')
    # rajout des variables d'evolution des deux dernieres commandes
    last_2_order_df = test.rolling(3).agg({'delay':'mean','sales_net':'mean','quantity':'mean','branch_id':'mean','product_id':'mean'})
    test = test.join(last_2_order_df,rsuffix='_2_last')
    # rajout des variables d'evolution des 5 dernieres commandes
    last_5_order_df = test.rolling(6).agg({'delay':'mean','sales_net':'mean','quantity':'mean','branch_id':'mean','product_id':'mean'})
    test = test.join(last_5_order_df,rsuffix='_5_last')
    # rajout des variables d'evolution des 5 dernieres commandes
    last_10_order_df = test.rolling(11).agg({'delay':'mean','sales_net':'mean','quantity':'mean','branch_id':'mean','product_id':'mean'})
    test = test.join(last_10_order_df,rsuffix='_10_last')
    # selection de seulement la ligne comportant la date finale par client
    test = test.merge(test.groupby('client_id').date_order.max().reset_index(), on=['client_id','date_order'])
    # suppression des colonnes inutiles
    final_df = client_df.reset_index().join(test, rsuffix='_bis').drop(['client_id_bis','date_order','delay','sales_net','quantity','branch_id','product_id'],axis=1)

    print('---Calcul des évolutions par client---')
    # calcul des evolutions pour tous
    delay=['delay_last','delay_2_last','delay_5_last','delay_10_last']
    for col in delay:
        final_df[col] = (final_df['delay_mean'] - final_df[col])/final_df['delay_mean']
    quantity = ['quantity_last','quantity_2_last','quantity_5_last','quantity_10_last']
    for col in quantity:
        final_df[col] = (final_df['mean_quantity'] - final_df[col])/final_df['mean_quantity']
    sales = ['sales_net_last','sales_net_2_last','sales_net_5_last','sales_net_10_last']
    for col in sales:
        final_df[col] = (final_df['mean_sales_net'] - final_df[col])/final_df['mean_sales_net']
    branch = ['branch_id_last','branch_id_2_last','branch_id_5_last','branch_id_10_last']
    for col in branch:
        final_df[col] = (final_df['mean_nb_of_branch'] - final_df[col])/final_df['mean_nb_of_branch']
    product = ['product_id_last','product_id_2_last','product_id_5_last','product_id_10_last']
    for col in product:
        final_df[col] = (final_df['mean_nb_of_product'] - final_df[col])/final_df['mean_nb_of_product']
    util_df = final_df[['client_id']+delay+quantity+sales+branch+product+['label']]

    print('---Ajout des colonnes negative amount---')
    # ajout des colonnes has negative amount
    has_neg_df = df.groupby(['client_id','date_order']).agg({'sales_net':'min'})
    last_is_neg = has_neg_df.reset_index().rolling(2).sales_net.min()
    has_neg_df = has_neg_df.reset_index().join(last_is_neg, rsuffix='_last')
    last_2_is_neg = has_neg_df.rolling(3).sales_net.min()
    has_neg_df = has_neg_df.join(last_2_is_neg, rsuffix='_2_last')
    last_5_is_neg = has_neg_df.rolling(6).sales_net.min()
    has_neg_df = has_neg_df.join(last_5_is_neg, rsuffix='_5_last')
    has_neg_df = has_neg_df.merge(has_neg_df.groupby('client_id').date_order.max().reset_index(), on=['client_id','date_order'])
    has_neg_df = has_neg_df.drop('sales_net',axis=1)
    # conversion en 0 ou 1 suivant la présence d'un montant négatif ou non
    has_neg_df.sales_net_last = has_neg_df.sales_net_last.map(lambda x: 1 if x<=0 else 0)
    has_neg_df.sales_net_2_last = has_neg_df.sales_net_2_last.map(lambda x: 1 if x<=0 else 0)
    has_neg_df.sales_net_5_last = has_neg_df.sales_net_5_last.map(lambda x: 1 if x<=0 else 0)
    has_neg_df = has_neg_df.drop(['date_order'],axis=1)
    # ajout de la colone a util_df et renaming des colonnes
    util_df = util_df.join(has_neg_df,rsuffix='_bis').drop('client_id_bis',axis=1).rename({'sales_net_last_bis':'last_is_neg','sales_net_2_last_bis':'2_last_is_neg','sales_net_5_last_bis':'5_last_is_neg'},axis=1)

    print('---Préparation des données---')
    # préparation du dataset
    util_df = util_df.set_index('client_id')
    # on retire les clients n'ayant jamais commandé
    util_df = util_df.replace([np.inf, -np.inf], np.nan)
    util_df = util_df.dropna()
    return util_df

def churn_scaling(util_df):
    if util_df.shape[0]==0:
        raise ValueError('Le dataset ne contient aucun client dans le cluster selectionné')
    print('---Rescaling des données---')
    # rescaling des données
    scaler = StandardScaler()
    X = util_df.drop('label',axis=1)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=util_df.drop('label',axis=1).columns)
    return X

def churn_train(util_df,cluster):
    print('---Prediction & calcul de la matrice de confusion---')
    # prediction et affichage des matrices de confusion
    util_df_scaled = churn_scaling(util_df)
    X_train, X_test, y_train, y_test = train_test_split(util_df_scaled,util_df.label,test_size=0.2, shuffle=True)
    XGB = XGBClassifier(gamma=0.8, subsample=0.6, max_depth=7)
    print('---Construction et enregistrement du modèle---')
    XGB.fit(X_train, y_train)
    filename = 'modele/churn_model'+'clucter_'+str(cluster)+'.sav'
    joblib.dump(XGB, filename)

    predictions = XGB.predict(X_test)
    pred_prob = XGB.predict_proba(X_test)[:,-1]
    accuracy = accuracy_score(y_test, predictions)
    AUC = roc_auc_score(y_test, pred_prob)
    conf_mat = confusion_matrix(y_test, predictions)
    print('---resultats du train---')
    print("Accuracy XGB : {}".format(accuracy))
    print("AUC XGB : {}".format(AUC))
    print("Matrice de confusion : \n{}".format(conf_mat))
    return XGB
   

def predict(util_df,cluster):
    filename = 'modele/churn_model'+'clucter_'+str(cluster)+'.sav'
    util_df_scaled = churn_scaling(util_df)
    try:
        print('---Téléchargement du modèle---')
        XGB = joblib.load(filename)
    except:
        raise ValueError('Le modèle n\'est pas entrainé, vous devez entrainer le modèle avant de prédire')
    predictions = XGB.predict(util_df_scaled)
    churner = pd.DataFrame(predictions, index=util_df.client_id, columns=['prediction'])
    return churner.prediction.sort_values(ascending=False)

def labelize(cluster,df):
    client_df = df.groupby(['client_id']).agg({'date_order':'max'})
    client_df = client_df.rename({'date_order':'last_order_date'},axis=1)
    today = datetime(2019,10,22)
    # ajout de la colonne churn factor
    nb_order_df = df[['client_id','date_order']].groupby('client_id').agg({'date_order':pd.Series.nunique})
    nb_order_df = nb_order_df.rename({'date_order':'nb_order'},axis=1)
    client_window_df = df[['date_order','client_id']].groupby('client_id').date_order.max()-df[['date_order','client_id']].groupby('client_id').date_order.min()
    client_window_df = client_window_df.map(lambda x: x.days)
    order_freqency_df = client_window_df/(nb_order_df.nb_order-1)
    order_freqency_df = order_freqency_df.rename('order_frequency')
    order_freqency_df.loc[order_freqency_df==np.inf] = 0
    client_df = client_df.join(order_freqency_df)
    # retrait des clients ayant commandés une seule fois
    client_df=client_df.dropna()
    client_df = client_df[client_df.order_frequency>1]
    client_df['time_since_last_order'] = today-client_df.last_order_date
    client_df['time_since_last_order'] = client_df['time_since_last_order'].map(lambda x: x.days)
    client_df['churn_factor'] = client_df.time_since_last_order/client_df.order_frequency
    # Nous prenons la médiane du churn factor pour labelliser les clients selon deux catégories
    # suivant le cluster auquel ils appartiennent
    if cluster == 1:
        client_df['churn_factor'] = client_df['churn_factor'].map(lambda x: 1 if x>3.75 else 0)
    elif cluster == 0:
        client_df['churn_factor'] = client_df['churn_factor'].map(lambda x: 1 if x>3.5 else 0)
    elif cluster == 2:
        client_df['churn_factor'] = client_df['churn_factor'].map(lambda x: 1 if x>5 else 0)
    df = df.merge(client_df, on='client_id')
    df = df.drop(['last_order_date','time_since_last_order','order_frequency'],axis=1)
    df = df.rename({'churn_factor':'label'},axis=1)
    return df

def optimize_model(util_df,cluster):
    filename = 'modele/churn_model'+'clucter_'+str(cluster)+'.sav'
    try:
        print('---Téléchargement du modèle---')
        XGB = joblib.load(filename)
    except:
        raise ValueError('Le modèle n\'est pas entrainé, vous devez entrainer le modèle avant de l\'optimiser')
    
    param_grid = {
        'gamma': [0.8,1,1.2],
        'subsample': [0.6,0.8,1],
        'max_depth': [6,7,8]
        }

    gs = GridSearchCV(XGB, param_grid, cv=5)
    gs.fit(util_df.drop('label',axis=1), util_df.label)
    params_opt = gs.best_params_
    score_opt = gs.best_score_

    print("params XGBoost optimisés : {}".format(params_opt))
    print("score optimisé : {}".format(score_opt))