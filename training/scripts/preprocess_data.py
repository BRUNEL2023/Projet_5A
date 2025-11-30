import numpy as np
import pandas as pd
import pywt
import librosa
import glob
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fonction pour charger les signaux à partir des fichiers CSV dans un dossier
def load_signals_from_folder(folder_path):
    signals = []
    for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
        signal_data = pd.read_csv(file_path)
        signals.append(signal_data)
    return signals

# Filtrage du signal avec un filtre passe-bande
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Application du filtre passe-bande
def apply_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=1000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Fonction pour extraire les caractéristiques DWT
def extract_dwt_features(signal, max_features=100):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    features = np.concatenate([c.flatten() for c in coeffs])
    features = features[:max_features]
    if len(features) < max_features:
        features = np.pad(features, (0, max_features - len(features)), 'constant')
    return features

# Fonction pour extraire les caractéristiques MFCC
def extract_mfcc_features(signal, sr=1000, max_mfcc=13):
    signal = np.array(signal).flatten()
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=max_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Fonction pour normaliser les caractéristiques
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Fonction principale pour prétraiter les données
def preprocess_data(data_dir, save_dir):
    all_features = []
    all_labels = []

    for gesture_label in os.listdir(data_dir):
        gesture_folder = os.path.join(data_dir, gesture_label)
        
        if not os.path.isdir(gesture_folder):
            continue

        print(f"Traitement du geste : {gesture_label}")

        signals = load_signals_from_folder(gesture_folder)

        for sig in signals:
            raw_signal = sig.iloc[:, 1:7].values.flatten()

            filtered_signal = apply_bandpass_filter(raw_signal)

            dwt_features = extract_dwt_features(filtered_signal)
            mfcc_features = extract_mfcc_features(filtered_signal)

            combined_features = np.concatenate((dwt_features, mfcc_features))

            all_features.append(combined_features)
            all_labels.append(gesture_label)

    # Normalisation des caractéristiques extraites
    all_features = normalize_features(all_features)

    # Séparation en train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

    # Sauvegarde des datasets
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(save_dir, 'amg_dataset_train.csv')
    test_path  = os.path.join(save_dir, 'amg_dataset_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset train sauvegardé à {train_path}")
    print(f"Dataset test sauvegardé à {test_path}")

# Exécution du pipeline
DATA_DIR = 'C:/Users/PC/Documents/Projet 5A/data'  # Remplacer par le bon chemin
SAVE_DIR = 'C:/Users/PC/Documents/a_PROJET_5A/Projet_5A/data/preprocessed'

preprocess_data(DATA_DIR, SAVE_DIR)





"""
import numpy as np
import pandas as pd
import pywt
import librosa
import glob
import os
from scipy.signal import butter, filtfilt

# Fonction pour charger les signaux à partir des fichiers CSV dans un dossier
def load_signals_from_folder(folder_path):
    signals = []
    for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
        # Charger chaque fichier CSV
        signal_data = pd.read_csv(file_path)
        signals.append(signal_data)
    return signals

# Filtrage du signal avec un filtre passe-bande
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Application du filtre passe-bande
def apply_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=1000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Fonction pour extraire les caractéristiques DWT
def extract_dwt_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    features = np.concatenate([coeff.flatten() for coeff in coeffs])  # Aplatissement des coefficients
    return features

# Fonction pour extraire les caractéristiques MFCC
def extract_mfcc_features(signal, sr=1000):
    # Convertir signal en tableau NumPy
    signal = np.array(signal)
    
    # Librosa attend un signal sous forme de tableau NumPy, donc il faut s'assurer de ce type
    if signal.ndim == 1:  # Vérifier si le signal est un vecteur 1D
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    else:
        # Si le signal est multidimensionnel (plusieurs canaux), on le convertit en 1D
        signal = signal.flatten()  # Aplatir le signal en un vecteur 1D
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    # Calculer la moyenne des MFCC sur le temps (axis=1)
    mfcc_features = np.mean(mfcc, axis=1)
    return mfcc_features

# Fonction principale pour prétraiter les données
def preprocess_data(data_dir):
    all_features = []
    all_labels = []
    
    for gesture_label in os.listdir(data_dir):
        gesture_folder = os.path.join(data_dir, gesture_label)
        
        if os.path.isdir(gesture_folder):
            signals = load_signals_from_folder(gesture_folder)
            
            for signal in signals:
                filtered_signal = apply_bandpass_filter(signal.iloc[:, 1:7].values.flatten())
                dwt_features = extract_dwt_features(filtered_signal)
                mfcc_features = extract_mfcc_features(filtered_signal)
                
                combined_features = np.concatenate((dwt_features, mfcc_features))
                all_features.append(combined_features)
                all_labels.append(gesture_label)

    features_df = pd.DataFrame(all_features)
    labels_df = pd.DataFrame(all_labels, columns=['label'])
    
    dataset = pd.concat([features_df, labels_df], axis=1)
    
    dataset.to_csv('C:/Users/PC/Documents/a_PROJET_5A/Projet_5A/data/preprocessed/amg_dataset.csv', index=False)
    print("Dataset prétraité et sauvegardé avec succès.")

preprocess_data('C:/Users/PC/Documents/Projet 5A/data')
"""