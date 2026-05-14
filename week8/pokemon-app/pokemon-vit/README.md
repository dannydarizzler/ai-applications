---
library_name: transformers
license: apache-2.0
base_model: google/vit-base-patch16-224
tags:
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: pokemon-vit
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pokemon-vit

This model is a fine-tuned version of [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9193
- Accuracy: 0.8

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 10   | 1.7264          | 0.2414   |
| 1.6383        | 2.0   | 20   | 1.5261          | 0.3448   |
| 1.6383        | 3.0   | 30   | 1.3662          | 0.4828   |
| 1.1842        | 4.0   | 40   | 1.2184          | 0.5862   |
| 1.1842        | 5.0   | 50   | 1.1141          | 0.6552   |
| 0.9387        | 6.0   | 60   | 1.0422          | 0.7931   |
| 0.9387        | 7.0   | 70   | 0.9947          | 0.7931   |
| 0.8110        | 8.0   | 80   | 0.9617          | 0.7931   |
| 0.8110        | 9.0   | 90   | 0.9447          | 0.7931   |
| 0.7525        | 10.0  | 100  | 0.9382          | 0.7931   |


### Framework versions

- Transformers 5.5.4
- Pytorch 2.11.0
- Datasets 4.8.4
- Tokenizers 0.22.2
