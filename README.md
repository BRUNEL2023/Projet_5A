# Projet_5A
Controle d'une prothèse de la main en temps réel avec l'IA
## Construction du Dataset
Le dataset utilisé pour entraîner le modèle a été créé à partir des signaux AMG collectés par le bracelet myo-acoustique. 
Après avoir nettoyé et filtré les signaux bruts, des caractéristiques telles que les coefficients DWT et les MFCC ont été extraites.
Le dataset est ensuite structuré sous forme de tableau, où chaque ligne correspond à une fenêtre temporelle et les colonnes contiennent les caractéristiques extraites, tandis que le label correspond au geste effectué.
