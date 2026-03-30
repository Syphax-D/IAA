import os
from PIL import Image, ImageFilter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib

def buildSampleFromPath(path1, path2):
    h = 64
    l = 64
    S = []
    
    for path, target in [(path1, 1), (path2, -1)]:
        for filename in os.listdir(path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(path, filename)
            
            img = Image.open(file_path).convert("HSV")
            img_resized = resizeImage(img, h, l)
            histo_original = computeHisto(img_resized)
            
            S.append({
                "name_path": file_path,
                "X_histo": histo_original,
                "y_true_class": target,
                "y_predicted_class": None,
                "is_augmented": False
            })
            
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flipped_resized = resizeImage(img_flipped, h, l)
            histo_flipped = computeHisto(img_flipped_resized)
            
            S.append({
                "name_path": file_path + "_flipped",
                "X_histo": histo_flipped,
                "y_true_class": target,
                "y_predicted_class": None,
                "is_augmented": True
            })

            img.close()
            img_flipped.close()
            
    return S

def resizeImage(i, h, l):
    return i.resize((l, h))

def computeHisto(i):
    width, height = i.size
    
    # Caractéristiques Couleur (Spatialisation : Haut, Milieu, Bas)
    h_tiers = height // 3
    
    bloc_haut = i.crop((0, 0, width, h_tiers))
    bloc_milieu = i.crop((0, h_tiers, width, 2 * h_tiers))
    bloc_bas = i.crop((0, 2 * h_tiers, width, height))
    
    taille_paquet = 8
    
    def process_histo(bloc):
        histo = bloc.histogram()
        histo_reduit = [sum(histo[j:j+taille_paquet]) for j in range(0, len(histo), taille_paquet)]
        total_pixels = sum(histo_reduit)
        if total_pixels == 0:
            return histo_reduit
        return [val / total_pixels for val in histo_reduit]

    features = []
    features.extend(process_histo(bloc_haut))
    features.extend(process_histo(bloc_milieu))
    features.extend(process_histo(bloc_bas))
    
    img_gray = i.convert("L")
    img_edges = img_gray.filter(ImageFilter.FIND_EDGES)
    
    histo_edges = img_edges.histogram()
    
    taille_paquet_edges = 16
    histo_edges_reduit = [sum(histo_edges[j:j+taille_paquet_edges]) for j in range(0, len(histo_edges), taille_paquet_edges)]
    
    total_edges = sum(histo_edges_reduit)
    if total_edges > 0:
        histo_edges_reduit = [val / total_edges for val in histo_edges_reduit]
        
    features.extend(histo_edges_reduit)
    
    return features

def fitFromHisto(S, algo):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]
    
    if algo["name"].startswith("RandomForest"):
        model = RandomForestClassifier(**algo["hyper_param"], random_state=42)
    elif algo["name"].startswith("SVM"):
        model = make_pipeline(StandardScaler(), SVC(**algo["hyper_param"], random_state=42))
    else:
        raise ValueError("Algorithme pas reconnu : " + algo["name"])
        
    model.fit(X, y)
    return model

def predictFromHisto(S, model):
    predictions = []
    for img in S:
        x = img["X_histo"]
        y_pred = model.predict([x])[0]
        img["y_predicted_class"] = y_pred
        predictions.append(y_pred)
    return predictions

def erreurempirique(S):
    y_true = [img["y_true_class"] for img in S]
    y_pred = [img["y_predicted_class"] for img in S]
    return 1 - accuracy_score(y_true, y_pred)

def crossValidationError(S, algo, k):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]

    if algo["name"].startswith("RandomForest"):
        model = RandomForestClassifier(**algo["hyper_param"], random_state=42)
    elif algo["name"].startswith("SVM"):
        model = make_pipeline(StandardScaler(), SVC(**algo["hyper_param"], random_state=42))
    else:
        raise ValueError("Algorithme pas reconnu : " + algo["name"])

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = cross_val_score(model, X, y, cv=kf)
    return 1 - np.mean(accuracies)

path1 = r"./Init/Mer" 
path2 = r"./Init/Ailleurs"
S = buildSampleFromPath(path1, path2)
print("Nombre d'images chargees :", len(S))

algos = [
    {
        "name": "RandomForest", 
        "hyper_param": {
            "n_estimators": 200,
            "max_depth": 3,
            "min_samples_leaf": 10
        }
    },
    {
        "name": "SVM (RBF, C=1)",
        "hyper_param": {
            "kernel": "rbf",
            "C": 1,
            "gamma": "scale"
        }
    }
]

for algo in algos:
    print(f"\nRésultats pour {algo['name']}")
    model = fitFromHisto(S, algo)
    predictFromHisto(S, model)
    print(f"Erreur empirique  : {erreurempirique(S)*100:.2f} %")
    print(f"Erreur Cross-Val  : {crossValidationError(S, algo, k=5)*100:.2f} %")

# Décommentez seulement le modele que vous voulez sauvegarder
    #if algo["name"] == "RandomForest":
        #nom_fichier_modele = "modele_random_forest.joblib"
        #joblib.dump(model, nom_fichier_modele)
        #print(f"Modele sauvgardé sous: {nom_fichier_modele}")
    #if algo["name"] == "SVM (RBF, C=1)":
        #nom_fichier_modele = "modele_svm.joblib"
        #joblib.dump(model, nom_fichier_modele)
        #print(f"Modele sauvgardé sous: {nom_fichier_modele}")