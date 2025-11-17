# Projet_5A
Controle d'une prothèse de la main en temps réel avec l'IA
## Construction du Dataset
Le dataset utilisé pour entraîner le modèle a été créé à partir des signaux AMG collectés par le bracelet myo-acoustique. 
Après avoir nettoyé et filtré les signaux bruts, des caractéristiques telles que les coefficients DWT et les MFCC ont été extraites.
Le dataset est ensuite structuré sous forme de tableau, où chaque ligne correspond à une fenêtre temporelle et les colonnes contiennent les caractéristiques extraites, tandis que le label correspond au geste effectué.

## Prétraitement des données

Le script `preprocess_data.py` est utilisé pour charger, filtrer et extraire les caractéristiques des signaux myo-acoustiques (AMG) collectés. Le prétraitement inclut :

1. ## Chargement des données 

 Les signaux sont chargés à partir des fichiers CSV présents dans les sous-dossiers correspondants à chaque geste de la main.
2. ## Filtrage des signaux
    
Seules les colonnes des signaux de 1 à 6 sont utilisées, correspondant aux entrées des capteurs de la main.
3. ## Extraction des caractéristiques :
   - DWT (Décomposition en ondelettes): Les signaux sont décomposés en plusieurs niveaux de fréquence pour extraire des caractéristiques temporelles et fréquentielles.
   - MFCC (Mel-Frequency Cepstral Coefficients)
    Des caractéristiques spectrales sont extraites des signaux, similaires à celles utilisées en traitement du signal audio.
4. ## Structuration du dataset 
 Les caractéristiques extraites sont combinées dans un fichier CSV, et ce dataset est sauvegardé dans le répertoire `data/preprocessed/` sous le nom `amg_dataset.csv`.
