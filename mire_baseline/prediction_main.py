import json
import torch
from PIL import Image
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import torch

# Charger le modèle
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",          # The model to load (Qwen2-VL-7B-Instruct)
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # Important pour certains modèles comme Qwen 
)

# Charger le processeur
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# ===============
# =============== (  1  ) =================================

# Charger les données de test depuis un fichier JSON
def load_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================== (  2  ) =================================

# Fonction de traitement des données
def process_test_data(test_data, output_file_path):
    success_count = 0  # Compteur pour les traitements réussis

    for i, item in enumerate(test_data, start=1):
        instruction = item['instruction']
        images = item['image']  # Liste d'images
        id = item['id']

        # Charger toutes les images existantes
        image_objects = []
        for image_name in images:
            image_path = f"MIRE-main/Images-scenes/{image_name}"
            if os.path.exists(image_path):
                image_objects.append(Image.open(image_path))
            else:
                #print(f"L'image {image_path} n'existe pas. Elle sera ignorée.")
                continue

        # Si aucune image n'est valide, ignorer l'élément
        if not image_objects:
            #print(f"Aucune image valide trouvée pour l'élément {id}. Ignoré.")
            continue

        # Créer le message d'entrée avec toutes les images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image} for image in image_objects
                ] + [{"type": "text", "text": instruction}],
            }
        ]

        # Préparation pour l'inférence
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Traitement des informations visuelles (images uniquement)
        image_inputs, _ = process_vision_info(messages)

        # Assurer que les entrées sont correctes
        if image_inputs is None:
            print(f"Erreur : Les entrées d'images sont None pour l'élément {id}.")
            continue

        # Préparer les entrées pour l'inférence
        inputs = processor(
            text=[instruction],  # Supposons que 'instruction' soit une chaîne de texte
            padding=True,
            return_tensors="pt",
        )

        # Vérification de la disponibilité du GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)

        try:
            # Génération de la prédiction
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Décodage de la sortie générée
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Ajouter le résultat dans le fichier JSONL
            with open(output_file_path, "a", encoding="utf-8") as f:
                result = {
                    'id': item['id'],
                    'predict': output_text[0]
                }
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")  # Nouvelle ligne pour chaque objet JSON

            # Incrémenter le compteur de traitements réussis
            success_count += 1

            # Si 1000 traitements réussis ont été effectués, arrêter la boucle
            if success_count >= 1000:
                print(f"Seuil de 1000 traitements réussis atteint. Arrêt du traitement.")
                break

        except Exception as e:
            print(f"Erreur lors de la génération pour l'élément {id} : {e}")

        print(f"{i}) Traitement terminé pour l'élément {id}")

    return output_file_path

# ============================== (  3  ) =================================

# Charger les données de test à partir du fichier JSON
file_path = "MIRE-main/data/test.json"
test_data = load_test_data(file_path)

# ============================== (  4  ) =================================

# Définir le chemin du fichier de sortie JSONL
output_file_path = "MIRE-main/Output/results.jsonl"

# ============================== (  5  ) =================================

# Traitement du fichier de test et génération du fichier JSONL
output_file = process_test_data(test_data, output_file_path)

print(f"Les résultats ont été enregistrés dans : {output_file}")