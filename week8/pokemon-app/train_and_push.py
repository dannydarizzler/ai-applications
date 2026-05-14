import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
import evaluate
from huggingface_hub import login
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "Danydarizzler"
MODEL_REPO = f"{HF_USERNAME}/pokemon-vit"
MODEL_CKPT = "google/vit-base-patch16-224"
DATA_DIR = "../transferlearning_with_custom_data/data/pokemon"

print("Logging in to HuggingFace...")
login(token=HF_TOKEN)

print("Loading dataset...")
dataset = load_dataset("imagefolder", data_dir=DATA_DIR)

label_names = dataset["train"].features["label"].int2str
labels = dataset["train"].unique("label")
print(f"{len(labels)} classes:", [label_names(l) for l in labels])

split = dataset["train"].train_test_split(test_size=0.15, seed=42)
our_dataset = DatasetDict({
    "train":      split["train"],
    "validation": split["test"],
    "test":       dataset["test"],
})

label2id = {label_names(c): c for c in labels}
id2label = {c: label_names(c) for c in labels}

print("Loading processor...")
processor = AutoImageProcessor.from_pretrained(MODEL_CKPT)

def transforms(batch):
    batch["image"] = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(batch["image"], return_tensors="pt")
    inputs["labels"] = [label2id[label_names(y)] for y in batch["label"]]
    return inputs

processed_dataset = our_dataset.with_transform(transforms)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels":       torch.tensor([x["labels"] for x in batch]),
    }

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

print("Loading model...")
model = ViTForImageClassification.from_pretrained(
    MODEL_CKPT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

for name, p in model.named_parameters():
    if not name.startswith("classifier"):
        p.requires_grad = False

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,} | Trainable: {trainable:,}")

training_args = TrainingArguments(
    output_dir="./pokemon-vit",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    num_train_epochs=10,
    learning_rate=3e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    hub_model_id=MODEL_REPO,
    hub_token=HF_TOKEN,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    processing_class=processor,
)

print("Starting training...")
trainer.train()

print("Evaluating on test set...")
results = trainer.evaluate(processed_dataset["test"])
print("Test results:", results)

print(f"Pushing model to HuggingFace: {MODEL_REPO}")
trainer.push_to_hub("Final model upload")
print(f"Done! Model available at: https://huggingface.co/{MODEL_REPO}")
