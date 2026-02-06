# 🍩 Stable Diffusion : Setup Simpsons Réaliste (LoRA + Photorealism)

Ce document contient les réglages et les prompts optimisés pour le modèle **LoRA Simpsons** mixé avec un modèle de base **Réaliste**.

## ⚙️ Configuration Globale

### 🚫 Negative Prompt (À utiliser systématiquement)
> `cartoon, drawing, anime, 2d, illustration, painting, yellow skin, flat colors, cel shading, low quality, blurry, deformed, distorted, bad anatomy, ugly, extra fingers, mutations`

### 🔧 Réglage du LoRA (Strength / Scale)
| Valeur | Effet |
| :--- | :--- |
| **0.4** | Très humain, traits des Simpsons subtils. |
| **0.6** | **Point d'équilibre idéal** (Reconnaissable mais humain). |
| **0.8** | Style plus marqué, risque de déformations ou peau jaune. |

---

## 📋 Bibliothèque de Prompts

| Catégorie | Sujet | Prompt (Anglais) |
| :--- | :--- | :--- |
| **Portrait** | **Homer** | `raw photo of Homer Simpson, sks simpson style, stubble beard texture, looking at camera, tired eyes, office fluorescent lighting, hyper realistic, 8k, highly detailed skin pores, shot on Canon 5D` |
| **Portrait** | **Marge** | `portrait of Marge Simpson, sks simpson style, tall blue beehive hairstyle made of real hair, wearing red pearl necklace, soft sunlight, suburban kitchen background, depth of field, masterpiece, photorealistic` |
| **Portrait** | **Mr. Burns** | `close up of Mr. Burns, sks simpson style, extremely wrinkled skin, liver spots, evil smile, steepled fingers, dark office background, ominous dramatic lighting, sharp focus, 8k` |
| **Portrait** | **Moe** | `Moe Szyslak, sks simpson style, sweaty face, unkempt hair, holding a dirty rag, inside a dim tavern, neon sign in background, bokeh, gritty texture, cinema look` |
| **Scène** | **Bart** | `full body shot of Bart Simpson, sks simpson style, holding a skateboard, wearing grunge street clothes, sunset lighting, golden hour, urban street background, lens flare, intricate details, 8k` |
| **Scène** | **Ned Flanders** | `Ned Flanders, sks simpson style, perfect mustache, glasses, wearing green sweater, overly happy smile, standing in a perfect garden, sunny day, sharp focus, high definition` |
| **Scène** | **Barney** | `Barney Gumble, sks simpson style, sitting at a bar, messy hair, 5 o'clock shadow, glass of beer, melancholic lighting, film grain, realistic texture` |
| **Scène** | **Lisa** | `Lisa Simpson playing saxophone, sks simpson style, jazz club atmosphere, smoke, stage spotlight, brass texture, concentrated expression, cinematic composition` |
| **Style** | **Sitcom 90s** | `Homer Simpson and Marge Simpson sitting on a couch, sks simpson style, 1990s sitcom screencap, vhs effect, flash photography, realistic living room, tv show promo` |
| **Style** | **Horreur** | `Sideshow Bob, sks simpson style, crazy wild hair, holding a knife, dark rainy alleyway, dramatic rim lighting, horror movie atmosphere, hyperdetailed, scary` |
| **Style** | **GTA Style** | `Fat Tony, sks simpson style, wearing a suit, smoking a cigar, gangster vibe, digital painting combined with photorealism, sharp shadows, vibrant colors, masterpiece` |
| **Autre** | **Krusty** | `Krusty the Clown, sks simpson style, depressing makeup, smoking a cigarette, backstage dressing room, realistic face paint texture, tired expression` |
| **Autre** | **Chef Wiggum** | `Chief Wiggum, sks simpson style, eating a donut, wearing police uniform, sitting in a patrol car, realistic fabric texture, police lights reflection` |
| **Autre** | **Willie** | `Groundskeeper Willie, sks simpson style, angry expression, red hair and beard, wearing a kilt, muscles, holding a shovel, schoolyard background, hdr` |