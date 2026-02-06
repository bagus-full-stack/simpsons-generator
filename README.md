# 🍩 Simpsons AI Generator

![Project Status](https://img.shields.io/badge/Status-Completed-success) ![Stack](https://img.shields.io/badge/Stack-FullStack-blue) ![AI Model](https://img.shields.io/badge/AI-Stable%20Diffusion-orange) ![License](https://img.shields.io/badge/License-MIT-green)

Une application Full-Stack complète permettant de générer des images dans le style des **Simpsons** en utilisant l'intelligence artificielle générative.

Le projet combine un **Backend Python (FastAPI)** puissant utilisant Stable Diffusion (LoRA, LCM, ControlNet, Inpainting) et un **Frontend moderne (Next.js)** offrant une expérience utilisateur fluide avec galerie, caméra et éditeur graphique.

---

## ✨ Fonctionnalités Principales

### 🎨 Modes de Génération

1.  **🔵 Texte-vers-Image :** Générez des personnages Simpsons à partir d'une simple description textuelle.
2.  **🟣 Image-vers-Image (Img2Img) :** Transformez vos photos (upload ou selfie webcam) en personnages jaunes tout en gardant les traits du visage.
3.  **🔴 Structure (ControlNet Canny) :** Capturez les contours d'un objet ou d'une pose spécifique pour forcer l'IA à respecter la composition exacte de l'image source.
4.  **🟢 Édition (Inpainting) :** Utilisez la "Gomme Magique" pour redessiner une partie spécifique de l'image (ex: ajouter un chapeau, changer la coiffure) sans toucher au reste.

### 🚀 Performance & UX

* **⚡ Mode Turbo (LCM-LoRA) :** Génération ultra-rapide (~1 seconde) pour des itérations rapides.
* **📷 Caméra Intégrée :** Prenez des photos directement depuis l'interface (Mobile & Desktop).
* **🖼️ Galerie Complète :** Historique de session persistant, pagination, zoom HD, téléchargement et suppression.
* **🖌️ Éditeur Canvas :** Pinceau ajustable avec curseur précis pour créer des masques de retouche.

---

## 🛠️ Stack Technique

### Backend (Python)
* **Framework :** FastAPI (API REST asynchrone).
* **Core IA :** PyTorch (CUDA), Diffusers (Hugging Face).
* **Modèles :**
    * Stable Diffusion v1.5 (Base).
    * LoRA personnalisé (Style Simpsons).
    * LCM-LoRA (Accélération Latent Consistency).
    * ControlNet Canny (Détection de contours).
* **Traitement d'image :** OpenCV, PIL, NumPy.

### Frontend (TypeScript)
* **Framework :** Next.js 14 (App Router).
* **Styling :** Tailwind CSS (Design Responsive & Thème Simpson).
* **Icônes :** Lucide React.
* **Interactions :** Canvas API (Dessin), MediaDevices API (Webcam).

---

## 📋 Prérequis

* **Système :** Windows (recommandé) ou Linux.
* **GPU :** Carte graphique NVIDIA recommandée (min 4Go VRAM) avec pilotes à jour.
* **Logiciels :**
    * **Python 3.10** (Impératif pour la compatibilité des dépendances IA).
    * **Node.js** (v18 ou supérieur).

---

## ⚙️ Installation

### 1. Backend (API)

Ouvrez un terminal dans le dossier `simpsonsGenerator` :

```bash
# 1. Créer un environnement virtuel avec Python 3.10
python -m venv .venv

# 2. Activer l'environnement (Windows)
.venv\Scripts\activate
# (Sur Mac/Linux : source .venv/bin/activate)

# 3. Installer PyTorch avec support CUDA (GPU)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Installer les dépendances du projet
pip install fastapi uvicorn python-multipart transformers accelerate peft diffusers opencv-python




Voici le contenu converti au format Markdown (`.md`), optimisé pour la clarté et la structure.

---

# Guide d'Utilisation

L'interface est divisée en **4 onglets intuitifs** :

* **Texte (Bleu) :** Entrez un prompt (ex: "Elon Musk"). Utilisez les suggestions pour aller plus vite.
* **Img2Img (Violet) :** Uploadez une image ou utilisez la Caméra. Réglez la **Ressemblance** (0.75 est un bon équilibre).
* **Structure (Rose) :** Idéal pour garder une pose ou une forme d'objet. L'IA détectera les contours via *Canny*.
* **Edit / Inpainting (Vert) :** Chargez une image. Dessinez en blanc sur la zone à modifier et décrivez le changement souhaité.

> ⚡ **Mode Turbo :** Activez le bouton jaune pour générer en moins d'une seconde (qualité légèrement inférieure).

---

## 📂 Structure du Projet

```plaintext
.
├── simpsonsGenerator/          # --- BACKEND ---
│   ├── .venv/                  # Environnement virtuel Python
│   ├── generated_simpsons/     # Stockage des images générées
│   ├── models/                 # Modèles téléchargés
│   ├── simpsons_lora_results/  # Dossier contenant votre LoRA
│   │   └── pytorch_lora_weights.safetensors
│   └── simpsonGeneratorAPI.py  # Point d'entrée de l'API
│
└── simpson-front/              # --- FRONTEND ---
    ├── public/
    ├── src/
    │   └── app/
    │       └── page.tsx        # Code principal de l'interface
    ├── package.json
    └── tailwind.config.ts

```

---

## 🔧 Dépannage (Troubleshooting)

| Erreur | Cause | Solution |
| --- | --- | --- |
| **"CUDA not available"** | Utilisation du processeur (CPU). | Réinstallez PyTorch avec la commande spécifique `--index-url`. |
| **"AttributeError: module 'mediapipe'..."** | Conflit de version. | `pip uninstall mediapipe controlnet_aux` et installez `opencv-python`. |
| **Image noire / mauvaise qualité (Turbo)** | Guidance Scale trop élevé pour le mode LCM. | Le code gère cela (passage à 1.5). Ne modifiez pas les réglages manuellement. |
| **Erreur Mémoire (OOM)** | Votre GPU sature. | Fermez les applications gourmandes ou redémarrez le script Python. |

---

## 📄 Licence

Ce projet est réalisé à des fins éducatives.

* **Stable Diffusion :** CreativeML Open RAIL-M.
* **Style Simpsons :** Fan art, usage non commercial recommandé.

---

Souhaitez-vous que j'ajoute des sections spécifiques pour l'installation des dépendances ou que je peaufine la mise en forme du tableau ?