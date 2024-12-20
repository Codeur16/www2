from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import torch
from PIL import Image
import os

# Charger le modèle
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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





# ============================== (  1  ) =================================

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
        images = item['image']
        id = item['id']

        # Charger l'image à partir du fichier
        image_path = "MIRE-main/Images-scenes/" + images[0]  # Assumer qu'il n'y a qu'une seule image
        #print(f"Image path: {image_path}")

        # Vérifier si l'image existe
        if not os.path.exists(image_path):
            #print(f" L'image {image_path} n'existe pas.")
            continue

        # Charger l'image
        image = Image.open(image_path)
        #print(f"Image object: {image}")

        # Créer le message d'entrée
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ],
            }
        ]

        # Vérification du processeur
        #print(f"Processor: {processor}")

        # Préparation pour l'inférence
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Traitement des informations visuelles (images uniquement)
        image_inputs, _ = process_vision_info(messages)

        # Vérification des entrées d'images
        #print(f"Image inputs: {image_inputs}")

        # Assurer que les entrées sont correctes
        if image_inputs is None:
            print("Erreur: Les entrées d'images sont None.")
            continue

        # Préparer les entrées pour l'inférence
        # Tester uniquement avec du texte
        inputs = processor(
            text=[instruction],  # Supposons que 'instruction' soit une chaîne de texte
            padding=True,
            return_tensors="pt",
        )
        # Vérification de la disponibilité du GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"Using device: {device}")
        inputs = inputs.to(device)
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Décodage de la sortie générée
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            #print(f"ID: {item['id']}, Output: {output_text[0]}")

            # Ajouter le résultat dans le fichier JSONL au fur et à mesure
            with open(output_file_path, "a", encoding="utf-8") as f:
                result = {
                    'id': item['id'],
                    'predict': output_text[0]
                }
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")  # Ajout d'une nouvelle ligne pour chaque objet JSON

            # Incrémenter le compteur de traitements réussis
            success_count += 1

            # Si 1000 traitements réussis ont été effectués, arrêter la boucle
            if success_count >= 1000:
                print(f"Seuil de 1000 traitements réussis atteint. Arrêt du traitement.")
                break

        except Exception as e:
            print(f"Erreur lors de la génération : {e}")
        
        print(f" {i} ) Traitement terminé pour l'image {id}")

    return output_file_path


# ============================== (  3  ) =================================

# Charger les données de test à partir du fichier JSON
file_path = "MIRE-main/data/test1.json"
test_data = load_test_data(file_path)


# ============================== (  4  ) =================================

# Définir le chemin du fichier de sortie JSONL
output_file_path = "MIRE-main/Output/results.jsonl"

# ============================== (  5  ) =================================
# Traitement du fichier de test et génération du fichier JSONL
output_file = process_test_data(test_data, output_file_path)

print(f"Les résultats ont été enregistrés dans : {output_file}")


























# ============================== (  1  ) =================================

# Charger les données de test depuis un fichier JSON
def load_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================== (  2  ) =================================

# Fonction de traitement des données
i=1
def process_test_data(test_data, output_file_path):
    results = []  # Liste pour stocker les résultats

    for item in test_data:
        instruction = item['instruction']
        images = item['image']
        id = item['id']
        
        # Charger l'image à partir du fichier
        image_path = "/content/drive/MyDrive/Colab Notebooks/www challenge/test_round1/images/" + images[0]  # Assumer qu'il n'y a qu'une seule image
        #print(f"Image path: {image_path}")
        
        # Vérifier si l'image existe
        if not os.path.exists(image_path):
            print(f"Erreur: L'image {image_path} n'existe pas.")
            continue
        
        # Charger l'image
        image = Image.open(image_path)
        #print(f"Image object: {image}")

        # Créer le message d'entrée
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ],
            }
        ]
        
        # Vérification du processeur
        #print(f"Processor: {processor}")

        # Préparation pour l'inférence
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Traitement des informations visuelles (images uniquement)
        image_inputs, _ = process_vision_info(messages)
        

        # Assurer que les entrées sont correctes
        if image_inputs is None:
            print("Erreur: Les entrées d'images sont None.")
            continue
        
        # Préparer les entrées pour l'inférence
        # Tester uniquement avec du texte
        inputs = processor(
            text=[instruction],  # Supposons que 'instruction' soit une chaîne de texte
            padding=True,
            return_tensors="pt",
        )
        # Vérification de la disponibilité du GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"Using device: {device}")
        inputs = inputs.to(device)
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Décodage de la sortie générée
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            #print(f"ID: {item['id']}, Output: {output_text[0]}")

            # Ajouter les résultats dans la liste
            results.append({
                'id': item['id'],
                'predict': output_text[0]
            })
        except Exception as e:
            print(f"Erreur lors de la génération : {e}")
        print(f" {i} ) Traitement terminé pour l'image {id}")

    # Sauvegarder les résultats dans un fichier JSONL
    with open(output_file_path, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")  # Ajout d'une nouvelle ligne pour chaque objet JSON

    return output_file_path


# ============================== (  3  ) =================================

# Charger les données de test à partir du fichier JSON
file_path = "/content/drive/MyDrive/Colab Notebooks/www challenge/test_round1/test1_item.json"
test_data = load_test_data(file_path)


# ============================== (  4  ) =================================

# Définir le chemin du fichier de sortie JSONL
output_file_path = "/content/drive/MyDrive/Colab Notebooks/www challenge/test_round1/results.jsonl"

# ============================== (  5  ) =================================
# Traitement du fichier de test et génération du fichier JSONL
output_file = process_test_data(test_data, output_file_path)

print(f"Les résultats ont été enregistrés dans : {output_file}")