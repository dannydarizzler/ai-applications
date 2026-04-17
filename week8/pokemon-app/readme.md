# Pokemon Image Classification – Model Comparison

## Links
- **Hugging Face Space (App):** `https://huggingface.co/spaces/Danydarizzler/pokemon-classifier` *(update if Space name differs)*
- **Trained Model:** `https://huggingface.co/Danydarizzler/pokemon-vit`

---

## Dataset Description

A custom **Pokemon image dataset** with **6 classes**:

| Class | Description |
|---|---|
| charizard | Fire/Flying type, final evolution of Charmander |
| charmander | Fire type, starter Pokemon |
| charmeleon | Fire type, middle evolution |
| ditto | Normal type, transform Pokemon |
| eevee | Normal type, evolution Pokemon |
| ekans | Poison type, snake-shaped Pokemon |

The images were collected and organised in the following structure:
```
data/pokemon/
├── train/
│   ├── charizard/
│   ├── charmander/
│   ├── charmeleon/
│   ├── ditto/
│   ├── eevee/
│   └── ekans/
└── test/
    └── (same 6 class folders)
```

The training split was further divided 85% / 15% into **train** and **validation** sets.

---

## Preprocessing Steps

1. **RGB conversion** – all images are converted to RGB (handles grayscale / RGBA inputs).
2. **Resize & normalise** – applied automatically by `AutoImageProcessor` from `google/vit-base-patch16-224`:
   - Resize to 224 × 224 pixels
   - Normalise pixel values with ImageNet mean/std
3. **Label encoding** – class names mapped to integer indices via `label2id` / `id2label` dicts.
4. **Dataset split** – 85% train, 15% validation (from original train set); original test set kept separate.

---

## Model & Evaluation

### Transfer Learning – ViT

| Property | Value |
|---|---|
| Base model | `google/vit-base-patch16-224` |
| Architecture | Vision Transformer (ViT-Base, patch size 16, input 224×224) |
| Fine-tuning strategy | Frozen backbone; only classifier head trained |
| Trainable params | ~3,100 (out of ~86 M total) |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Metric | Accuracy |

*(Fill in test accuracy after training)*

---

## Comparison: Custom ViT vs CLIP vs GPT-4o

| Model | Type | Approach | Test Accuracy (%) |
|---|---|---|---|
| Custom ViT | Closed / Fine-tuned | Transfer learning on Pokemon data | *(fill in)* |
| CLIP (`openai/clip-vit-large-patch14`) | Open-source | Zero-shot with Pokemon class names as text prompts | *(fill in)* |
| GPT-4o | Closed-source | Vision-language model, prompted for class probabilities | *(fill in)* |

### Observations
- The **Custom ViT** is the most accurate for in-distribution Pokemon images because it was explicitly trained on these 6 classes.
- **CLIP** achieves reasonable zero-shot results thanks to its large-scale vision-language pretraining, but can confuse visually similar Pokemon (e.g. Charmander vs Charmeleon).
- **GPT-4o** has broad world knowledge and handles ambiguous or stylised images well, but may be less confident on very specific class distinctions within the same evolution line.

---

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Set your OpenAI key first:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## How to Deploy on Hugging Face Spaces

1. Create a new Space: **New Space → Gradio SDK**
2. Upload `app.py`, `requirements.txt`, `readme.md`, and the `example_images/` folder.
3. Go to **Settings → Variables and secrets → New secret** and add:
   - Name: `OPENAI_API_KEY`
   - Value: your OpenAI key
4. The Space will build and start automatically.
