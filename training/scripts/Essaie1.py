import serial
import numpy as np
import joblib
import librosa
import time

# ============================
# CONFIG
# ============================
SR = 1024                  # fréquence d’échantillonnage
WINDOW_SIZE = 1024         # 1 seconde
HOP = 1024                 # pas = 1s (pas de recouvrement pour tester d’abord)
CHANNELS = 6

PORT = "COM13"              # adapte selon ton PC
BAUD = 921600

# === Paramètres MFCC identiques au preprocessing ===
N_MFCC = 20
N_MELS = 40
N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = N_FFT
FMIN = 0.0
FMAX = 128.0

# ============================
# CHARGEMENT DU MODELE
# ============================
svm = joblib.load(r"C:\Users\PC\Documents\a_PROJET_5A\Projet_5A\training\models\svm_mfcc.joblib")
scaler = joblib.load(r"C:\Users\PC\Documents\a_PROJET_5A\Projet_5A\training\models\svm_scaler.joblib")

print("Modèle chargé.")


# ============================
# FONCTION EXTRACTION FEATURES
# ============================
def features_mfcc_window(window):
    """
    window : shape (1024, 6)
    Retourne un vecteur (240,) comme dans l’entraînement SVM
    """
    feats = []

    for ch in range(CHANNELS):
        y = window[:, ch].astype(float)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            center=True,
            pad_mode="reflect",
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            dct_type=2,
            norm="ortho",
        )

        delta = librosa.feature.delta(mfcc, order=1)

        feats.extend(mfcc.mean(axis=1))
        feats.extend(delta.mean(axis=1))

    return np.array(feats)  # shape (240,)


# ============================
# OUVERTURE UART
# ============================
ser = serial.Serial(PORT, BAUD, timeout=0.01)
print(f"Port série {PORT} ouvert.")


# ============================
# BOUCLE TEMPS RÉEL
# ============================
buffer = np.zeros((WINDOW_SIZE, CHANNELS))
index = 0

print("=== Début acquisition temps réel ===")

while True:
    # Lecture du header
    head = ser.read(1)
    if head != b'\xAA':
        continue  # mauvais packet → on ignore

    # Lecture des 12 bytes suivants (6 x int16)
    data = ser.read(12)
    if len(data) != 12:
        continue

    # Conversion binaire → tableau Python
    values = np.frombuffer(data, dtype=np.int16)   # shape (6,)

    # Mise en buffer
    buffer[index] = values
    index += 1

    # Si une fenêtre complète est remplie
    if index >= WINDOW_SIZE:
        window = buffer.copy()

        # Extraction MFCC
        feat = features_mfcc_window(window)

        # Normalisation
        feat_scaled = scaler.transform([feat])

        # Prédiction
        pred = svm.predict(feat_scaled)[0]

        # Affichage
        print("→ Geste prédit :", pred)

        # Remet à zéro pour la prochaine fenêtre
        index = 0
