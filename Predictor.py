import os
import joblib
from PIL import Image, ImageFilter

class ImagePredictor:
    def __init__(self, model_path):
    
        self.model = joblib.load(model_path)
        self.h = 64
        self.l = 64

    def resizeImage(self, i):
        return i.resize((self.l, self.h))

    def computeHisto(self, i):
        width, height = i.size
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

    def predict_image(self, image_path):
        
        img = Image.open(image_path).convert("HSV")
        img_resized = self.resizeImage(img)
        features = self.computeHisto(img_resized)
        
        prediction = self.model.predict([features])[0]
        img.close()
        
        return prediction

    def predict_folder(self, folder_path, output_txt_path):

        if not os.path.exists(folder_path):
            print(f"Le dossier '{folder_path}' n'existe pas.")
            return

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                file_path = os.path.join(folder_path, filename)
                try:
                    prediction = self.predict_image(file_path)
                    
                    pred_str = "+1" if prediction == 1 else str(prediction)
                    
                    ligne = f"{filename} {pred_str}\n"
                    f.write(ligne)
                    
                except Exception as e:
                    print(f"Erreur lors de la prédiction pour {filename} : {e}")
                    
        print(f"Les prédictions ont été sauvegardées avec succès dans : {output_txt_path}")

if __name__ == "__main__":
    
    nom_modele = "modele_svm.joblib"
    
    # collez le File path directement ici
    dossier_a_tester = ("C:/Users/Syssou/Downloads/Data CC2/Data CC2")
    fichier_resultat = "resultats_predictions_svm.txt"
    
    if os.path.exists(nom_modele):
        print("Chargement du prédicteur")
        predictor = ImagePredictor(nom_modele)
        print(f"Analyse des images du dossier '{dossier_a_tester}'")
        predictor.predict_folder(dossier_a_tester, fichier_resultat)
    else:
        print(f"Erreur: Le modèle '{nom_modele}' est introuvable.")
