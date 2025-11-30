## Projet 5A - Contrôle d'une Prothèse de la Main en Temps Réel avec l'IA
Introduction

Ce projet vise à contrôler une prothèse de main en temps réel en utilisant des signaux myo-acoustiques (AMG) collectés à partir d'un bracelet myo-acoustique. L'objectif est de décoder l'activité musculaire et d'utiliser l'intelligence artificielle (IA) pour classifier les gestes effectués par l'utilisateur.

Objectif

L'objectif est de créer un modèle d'IA capable de reconnaître les gestes de la main en temps réel, en utilisant des caractéristiques extraites des signaux AMG et d'appliquer ces informations pour contrôler la prothèse de manière fluide et précise.

## Construction du Dataset

Le dataset utilisé pour entraîner le modèle a été créé à partir des signaux AMG collectés par le bracelet myo-acoustique, qui contient des informations sur l'activité musculaire de l'utilisateur. Ce dataset est structuré de manière à contenir les caractéristiques extraites des signaux, permettant d'entraîner un modèle d'IA pour classifier les gestes.

Nettoyage des données :
Les signaux bruts ont été filtrés pour éliminer le bruit et les interférences.

Extraction des caractéristiques :
Les caractéristiques suivantes ont été extraites des signaux :

DWT (Décomposition en Ondelette) : Cette technique est utilisée pour extraire les caractéristiques temporelles et fréquentielles des signaux.

MFCC (Mel-Frequency Cepstral Coefficients) : Ces caractéristiques spectrales sont extraites des signaux pour capturer des informations similaires à celles utilisées en traitement du signal audio.

Structuration du dataset :
Après l'extraction des caractéristiques, le dataset est structuré sous forme de tableau :

Colonnes : Les caractéristiques extraites (DWT, MFCC) de chaque fenêtre temporelle.

Ligne : Chaque ligne représente un échantillon de signal correspondant à une fenêtre temporelle d'un geste spécifique.

Label : Le label de chaque ligne correspond au geste effectué par l'utilisateur (par exemple, "fine_pinch", "tripod").

Le dataset est sauvegardé sous forme de fichiers CSV dans le répertoire data/preprocessed/ avec les noms :

amg_dataset_train.csv : Dataset d'entraînement.

amg_dataset_test.csv : Dataset de test.

Prétraitement des données

Le script preprocess_data.py est utilisé pour charger, nettoyer, filtrer, et extraire les caractéristiques des signaux myo-acoustiques. Ce prétraitement inclut les étapes suivantes :

1. Chargement des données

Les signaux sont chargés à partir des fichiers CSV présents dans les sous-dossiers, chaque sous-dossier correspondant à un geste de la main.

2. Filtrage des signaux

Seules les colonnes correspondant aux signaux des six capteurs (entrées de la main) sont utilisées pour extraire les caractéristiques. Ce prétraitement permet de ne conserver que les informations pertinentes.

3. Extraction des caractéristiques

Deux types de caractéristiques sont extraits des signaux :

DWT (Décomposition en ondelettes) : Les signaux sont décomposés en plusieurs niveaux de fréquence afin d'extraire des caractéristiques temporelles et fréquentielles.

MFCC (Mel-Frequency Cepstral Coefficients) : Des caractéristiques spectrales sont extraites, semblables à celles utilisées dans le traitement des signaux audio, pour capturer les variations des signaux sur le temps.

4. Structuration du dataset

Les caractéristiques extraites de chaque signal sont combinées dans un fichier CSV, avec chaque ligne représentant un signal et chaque colonne représentant une caractéristique extraite (DWT, MFCC). Ce dataset est ensuite sauvegardé dans le répertoire data/preprocessed/ sous le nom amg_dataset.csv.