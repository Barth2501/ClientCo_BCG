# Churn Detection algorithm

## Clustering de la clientèle
Dans un premier temps, nous avons effectué un clustering des clients dans le but de cibler des comportements de churn différents

### Démarche

Au vu du nombre important de client et comme nous ne connaissons pas à l'avance le nombre de cluster de clients, nous utiliserons un clustering en trois étapes :
- Clustering non hiérarchique, un kmeans, pour faire des gros cluster (500) de clients
- Clustering hierarchique avec critère de Ward afin de pouvoir observer sur le dendogramme le nombre de cluster idéal. Nous en choisissons 3
- Clustering non hierarchique, un deuxième kmeans, avec 3 clusters

### Visiualisation

Afin de visualiser au mieux la repartition de nos clusters, nous procédons à une analyse en composantes principales, pour les visualiser en 3D, selon les trois composantes principales de notre dataset.

Utilisez les fonctions show_dendogram et ACP pour observer respectivement le dendogramme et la visualisation 3D des données du dataset.

Nous observons donc que ces trois clusters sont caractérisées principalement par le canal d'achat, le nombre de produits différents commandés et le revenu par commande.

## Labelisation des données pour apprentissage

Afin de procéder à l'apprentissage du comportement d'un churner ous devons labelliser les données. Pour ce faire, nous calculons le churn-factor (durée depuis laquelle un client n'a pas commandé divisée par sa fréquence d'achat), nous nous plaçons à la médiane de cette valeur selon le cluster afin de séparer notre dataset en deux parties égales (0 : peu de risque de churner, 1 : fort risque de churner).
- Pour le cluster 0, le churn factor médian est de 3.5.
- Pour le 1, le churn factor médian est de 3.75.
- Pour le 2, il est de 5

## Algorithme de churn

Une fois les données d'entrainement labélisées, nous pouvons procéder à l'entrainement afin de faire apprendre à la machine le comportement que va avoir un client suscpetible de churn. Le modèle utilisé ici, est une classification avec un xgboost que nous pouvons optimiser via la fonction gridsearchCV de la fonction optimize_model.

### Entrainement du dataset

```
$ python main.py --dataset dataset.csv --cluster 0 --mode train
```
Cette commande aura pour but d'entrainer le modèle pour que celui sauvegarde les résultats dans le fichier approprié. L'utilisateur pourra par la suite prédire les données métier qu'il souhaite.

### Prediction du churner

```
$ python main.py --dataset dataset.csv --cluster 0 --mode predict
```
Cette commande aura pour but de charger le modèle pré-entrainé, puis de prédire la probabilité de churn des clients demandés. Une liste de client à voir en priorité sera renvoyée.