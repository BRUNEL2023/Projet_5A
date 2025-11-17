import os
import numpy as np
import pandas as pd
import pywt  # Pour la décomposition en ondelettes
import librosa  # Pour l'extraction des MFCC
import glob

# Fonction pour charger les signaux à partir des fichiers CSV dans un dossier
def load_signals_from_folder(folder_path):
    signals = []
    # Parcours des fichiers dans le dossier
    for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
        # Charger chaque fichier CSV
        signal_data = pd.read_csv(file_path)
        # Ajouter les signaux à la liste
        signals.append(signal_data)
    return signals

# Fonction pour filtrer les signaux de 1 à 6
def filter_signals(signals):
    filtered_signals = []
    for signal in signals:
        # Filtrer les signaux entre 1 et 6 (on suppose que les signaux sont numérotés)
        # Adaptation selon la structure exacte de votre fichier
        filtered_signals.append(signal.iloc[:, 1:7])  # En prenant seulement les colonnes 1 à 6
    return filtered_signals

# Fonction pour extraire les caractéristiques via la transformée en ondelettes (DWT)
def extract_dwt_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)  # Décomposition en 4 niveaux avec ondelette Daubechies
    features = np.concatenate([coeff.flatten() for coeff in coeffs])  # Aplatissement des coefficients
    return features

# Fonction pour extraire les caractéristiques MFCC
def extract_mfcc_features(signal, sr=16000):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extraire les 13 premiers MFCC
    mfcc_features = np.mean(mfcc, axis=1)  # Moyenne sur chaque coefficient pour obtenir une caractéristique par signal
    return mfcc_features

# Fonction principale pour prétraiter les données
def preprocess_data(data_dir):
    # Liste des signaux et des labels
    all_features = []
    all_labels = []
    
    # Parcours des sous-dossiers (chaque dossier correspond à un geste)
    for gesture_label in os.listdir(data_dir):
        gesture_folder = os.path.join(data_dir, gesture_label)
        
        if os.path.isdir(gesture_folder):
            # Charger les signaux pour ce geste
            signals = load_signals_from_folder(gesture_folder)
            
            # Filtrer et traiter chaque signal
            for signal in signals:
                # Filtrer les colonnes pour prendre les signaux de 1 à 6
                filtered_signal = filter_signals([signal])[0]
                
                # Extraire les caractéristiques de chaque signal
                dwt_features = extract_dwt_features(filtered_signal)
                mfcc_features = extract_mfcc_features(filtered_signal)
                
                # Combiner les caractéristiques DWT et MFCC
                combined_features = np.concatenate((dwt_features, mfcc_features))
                
                # Ajouter les caractéristiques et le label à la liste
                all_features.append(combined_features)
                all_labels.append(gesture_label)  # Le label du geste (nom du dossier)

    # Convertir en DataFrame pour faciliter l'enregistrement
    features_df = pd.DataFrame(all_features)
    labels_df = pd.DataFrame(all_labels, columns=['label'])
    
    # Combiner les caractéristiques et les labels
    dataset = pd.concat([features_df, labels_df], axis=1)
    
    # Sauvegarder le dataset structuré
    dataset.to_csv('data/preprocessed/amg_dataset.csv', index=False)
    print("Dataset prétraité et sauvegardé avec succès.")
    
# Appel de la fonction principale
preprocess_data('/mnt/data/unzipped_data/data')
