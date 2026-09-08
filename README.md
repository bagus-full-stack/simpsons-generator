# 🍩 Simpsons AI Generator

![Project Status](https://img.shields.io/badge/Status-Completed-success) ![Stack](https://img.shields.io/badge/Stack-FullStack-blue) ![AI Model](https://img.shields.io/badge/AI-Stable%20Diffusion-orange) ![License](https://img.shields.io/badge/License-MIT-green)

Une application Full-Stack complète permettant de générer des images dans le style des **Simpsons** en utilisant l'intelligence artificielle générative.

Le projet combine un **Backend Python (FastAPI)** puissant utilisant Stable Diffusion XL (LoRA, LCM, ControlNet, Inpainting) et un **Frontend moderne (Next.js)** offrant une expérience utilisateur fluide avec galerie, caméra et éditeur graphique.

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
    * Stable Diffusion XL 1.0 (Base, 1024x1024 natif).
    * LoRA personnalisé (Style Simpsons, entraîné sur base SDXL).
    * LCM-LoRA SDXL (Accélération Latent Consistency).
    * ControlNet Canny SDXL (Détection de contours).
* **Traitement d'image :** OpenCV, PIL, NumPy.

### Frontend (TypeScript)
* **Framework :** Next.js 16 (App Router).
* **Styling :** Tailwind CSS (Design Responsive & Thème Simpson).
* **Icônes :** Lucide React.
* **Interactions :** Canvas API (Dessin), MediaDevices API (Webcam).

---

## 📋 Prérequis

* **Système :** Windows (recommandé) ou Linux.
* **GPU :** Carte graphique NVIDIA recommandée (min 6-8Go VRAM pour SDXL) avec pilotes à jour.
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Installer les dépendances du projet
pip install -r requirements.txt

# 5. Copier la config d'exemple et l'ajuster si besoin (CORS, clé API, stockage...)
copy .env.example .env

# 6. Lancer l'API
uvicorn app.main:app --reload
```

Par défaut (`QUEUE_BACKEND=inline` dans `.env.example`), tout tourne dans un seul
process comme avant, sans dépendance externe.

### 2. Backend (Docker, alternative à l'étape 1)

Si vous préférez ne pas installer Python/CUDA en local (nécessite le NVIDIA
Container Toolkit pour exposer le GPU au conteneur) :

```bash
# API seule, file inline (même comportement que uvicorn ci-dessus)
docker compose up --build
```

Pour activer la file d'attente asynchrone (Redis + Postgres + worker séparé,
scalable indépendamment de l'API) :

```bash
docker compose --profile async up --build
```

Voir `.env.example` pour le détail de chaque variable (modèle, stockage
local/S3, modération, rate limit, Sentry...).

### Tests

```bash
pip install -r requirements-dev.txt
SKIP_MODEL_LOAD=1 pytest
```

---

## 🖥️ Frontend

Le frontend vit dans son propre dépôt Git, `simpson-front/` — cloné en tant que
dossier frère de celui-ci, pas comme sous-module (voir `.gitignore`) :

```bash
cd ../simpson-front
npm install
copy .env.local.example .env.local
npm run dev
```


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
├── app/                         # --- BACKEND (API) ---
│   ├── main.py                  # Point d'entrée : uvicorn app.main:app
│   ├── config.py                # Configuration (variables d'environnement)
│   ├── pipelines.py             # Chargement des pipelines Stable Diffusion
│   ├── generation.py            # Logique de génération (prompt, canny, inpaint...)
│   ├── jobs.py                  # Exécution d'un job (génération + modération + stockage)
│   ├── queue_backend.py         # File d'attente : inline (défaut) ou Redis/RQ
│   ├── worker.py                # Worker RQ séparé : python -m app.worker
│   ├── storage.py               # Stockage des images : local (défaut) ou S3
│   ├── security.py              # Auth par clé API + rate limiting
│   ├── moderation.py            # Filtre de contenu (prompt + NSFW image)
│   ├── db.py / models_db.py     # Base de données des jobs (SQLite ou Postgres)
│   └── routes/jobs.py           # Endpoints /jobs/*
│
├── tests/                       # Tests (SKIP_MODEL_LOAD=1, pas de GPU requis)
├── docker/Dockerfile            # Image API/worker (GPU, CUDA)
├── docker-compose.yml           # Stack locale (API, + Redis/Postgres/worker en option)
│
├── training/                    # Fine-tuning du LoRA (hors service de prod)
│   ├── train_simpsons_lora.py         # Script d'entraînement (remplace les notebooks)
│   ├── extract_frames_from_videos.py  # Dataset HD alternatif : frames extraites de tes propres épisodes (ffmpeg)
│   ├── simpsonGenerator*.ipynb  # Notebooks d'origine, conservés pour référence
│   └── prompts.md               # Bibliothèque de prompts d'entraînement
│
├── simpsons_lora_results/       # Poids LoRA entraînés, lus par app/pipelines.py
└── generated_simpsons/          # Stockage local des images (backend "local")
```

Le frontend (`simpson-front/`) est un **dépôt Git séparé**, cloné à côté de
celui-ci — pas un dossier de ce dépôt :

```plaintext
simpson-front/
├── app/
│   ├── page.tsx              # Interface (4 modes, galerie, caméra, éditeur)
│   └── lib/api.ts            # Client API (NEXT_PUBLIC_API_URL, polling des jobs)
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