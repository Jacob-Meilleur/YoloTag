import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO
import time

# --- CONFIGURATION ---
TAG_SIZE = 0.18  # Taille réelle du tag en MÈTRES (ex: 0.18 = 18cm)
WIDTH = 1280     # Largeur du flux vidéo
HEIGHT = 720    # Hauteur du flux vidéo
CAMERA_INDEX = 1 # Index de la caméra (0 pour caméra interne, 1 pour USB externe)
CONF_THRESHOLD = 0.75 # Seuil de confiance pour la détection YOLO

# Paramètres intrinsèques simplifiés [fx, fy, cx, cy]
# fx/fy : focales, cx/cy : centre optique. Idéalement, à obtenir via une calibration.
params = [1386, 1384, WIDTH // 2, HEIGHT // 2]

# Initialisation du détecteur d'AprilTags (famille 36h11 courante)
at_detector = Detector(families="tag36h11")
print("Étape A : Détecteur AprilTag initialisé avec succès !")

# Configuration de la capture vidéo OpenCV
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
# Utilisation du codec MJPG pour de meilleures performances en HD
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def calculate_homography(dst_pts):
    """
    Calcule la matrice d'homographie entre un carré unité et les coins détectés.
    dst_pts : Les 4 coins détectés par la caméra [(u1,v1), ...].
    """
    # Coordonnées du carré unité (référence interne de la bibliothèque)
    unit_src = np.array([
        [-1,  1], # Coin 0 : Bas-Gauche
        [ 1,  1], # Coin 1 : Bas-Droite
        [ 1, -1], # Coin 2 : Haut-Droite
        [-1, -1]  # Coin 3 : Haut-Gauche
    ])
    
    A = []
    for i in range(4):
        x, y = unit_src[i]
        u, v = dst_pts[i]
        # Équations de la Transformation Linéaire Directe (DLT)
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    
    # Résolution du système Ah = 0 par Décomposition en Valeurs Singulières (SVD)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] # La solution est le dernier vecteur propre
    
    # Mise en forme en matrice 3x3
    H = h.reshape((3, 3))
    
    # Normalisation pour que le dernier élément soit 1 (invariant d'échelle)
    return H / H[2, 2]

def get_real_coordinates(detection, parameters, tag_size):
    """
    Calcule la position X, Y, Z réelle du tag par rapport à la caméra.
    """
    corners = detection.corners
    H = calculate_homography(corners)
    
    fx, fy, cx, cy = parameters
    # Matrice de calibration intrinsèque de la caméra
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    
    # Inversion de la matrice K pour extraire les vecteurs de rotation et translation
    H_inv = np.linalg.inv(K) @ H
    r1 = H_inv[:, 0] # Vecteur de rotation 1
    r2 = H_inv[:, 1] # Vecteur de rotation 2
    t = H_inv[:, 2]  # Vecteur de translation
    
    # Calcul du facteur d'échelle pour convertir les unités arbitraires en mètres
    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    rel_x = t[0] / scale * (tag_size / 2)
    rel_y = t[1] / scale * (tag_size / 2)
    rel_z = t[2] / scale * (tag_size / 2)
    
    return rel_x, rel_y, rel_z

# Variable pour le suivi
person_count = 0

print("Démarrage du flux caméra...")
try:
    # Chargement du modèle YOLOv11 (version Nano pour la rapidité)
    # On force le CPU pour éviter les soucis de drivers GPU
    model = YOLO("yolo11n.pt", task='detect')
    model.to('cpu')
    print("Étape C : Modèle YOLO chargé !")
except Exception as e:
    print(f"Erreur lors du chargement de YOLO : {e}")

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    # Conversion en gris pour la détection AprilTag (plus rapide)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des AprilTags
    april_results = at_detector.detect(gray, 
                                     estimate_tag_pose=False, 
                                     camera_params=params, 
                                     tag_size=TAG_SIZE)
    
    # Détection d'objets avec YOLO (taille d'image réduite à 320 pour fluidité)
    YOLO_results = model(frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

    # Traitement des résultats AprilTag
    for r in april_results:  
        
        # Utilisation de la fonction d'homographie personnalisée
        rel_x, rel_y, rel_z = get_real_coordinates(r, params, TAG_SIZE)

        # Dessin des informations sur l'image
        cX, cY = int(r.center[0]), int(r.center[1])
        pos_text = f"X:{rel_x:.2f} Y:{rel_y:.2f} Z:{rel_z:.2f}"
        cv2.putText(frame, pos_text, (cX - 80, cY + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Tracé du contour du tag
        pts = r.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
    # Logique de détection de personne
    detected_this_frame = False
    for result in YOLO_results:
        # Vérifie si une classe 'person' est présente dans les boîtes englobantes
        if any(model.names[int(box.cls[0])] == 'person' for box in result.boxes):
            person_count += 1
        else:
            person_count = 0 # Réinitialise si personne n'est vu
        
    # Si une personne est vue de manière persistante (3 frames de suite)
    if person_count >= 3:
        print("CONFIRMÉ : Personne détectée.")
        person_count = 0 # Reset après confirmation

    # Affichage du rendu final
    cv2.imshow("Positionnement Relatif", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Quitter
        break
    

# Nettoyage des ressources
cap.release()
cv2.destroyAllWindows()