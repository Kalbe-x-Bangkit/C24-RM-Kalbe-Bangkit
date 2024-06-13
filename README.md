---
title: VQA Kalbe Bangkit
emoji: 🏆
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# Kalbe Farma - Visual Question Answering (VQA) for Medical Imaging

## Overview
This project addresses the challenge of accurate and efficient medical imaging analysis in healthcare, aiming to reduce human error and workload for radiologists. The proposed solution involves developing advanced AI models for Visual Question Answering (VQA) to assist healthcare professionals in analyzing medical images quickly and accurately. These models will be integrated into a user-friendly web application, providing a practical tool for real-world healthcare settings.  we provide fine-tune for medical imaging vqa task with unimodal model using VGG19-GPT2 and multimodal model using BLIP and idefics2

## Dataset
this project fine-tune pre-trained VLM model using these datasets :

rad-vqa dataset : https://huggingface.co/datasets/flaviagiammarino/vqa-rad

slake dataset : https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english

roco dataset : https://huggingface.co/datasets/mdwiratathya/ROCO-radiology



## Model Architecture

![Model Architecture](img/idefics2_architecture.png)


## Evaluation
the table below show evaluate result for fine-tuned model on medical dataset.

## Evaluation
The table below shows the evaluation results for the fine-tuned models on the medical datasets.

| Pretrain Model               | Dataset    | Epoch | Batch Size | Learning Rate | Optimizer | Train Loss | Val Loss | Accuracy  |
|------------------------------|------------|-------|------------|---------------|-----------|------------|----------|-----------|
| BLIP1                        | VQA-RAD    | 100   | 64         | 5.00E-05      | AdamW     | 9.65E-03   | 4.25E-01 | 43.68%    |
|salesforce/blip-capflit-large | VQA-RAD    | 100   | 64         | 5.00E-05      | RMSprop   | 1.82E-03   | 4.99E-01 | 41.02%    |
|                              | Path-VQA   | 100   | 64         | 5.00E-05      | RMSprop   | 9.42E-02   | 4.30E-01 | 19.33%    |
|                              | SLAKE      | 100   | 64         | 5.00E-05      | AdamW     | 9.99E-03   | 4.19E-01 | 45.67%    |
| Salesforce/blip-vqa-base     | VQA-RAD    | 100   | 64         | 5.00E-05      | AdamW     | 8.23E+00   | 5.78E-02 | 57.05%    |
|                              | Path-VQA   | 100   | 64         | 5.00E-05      | AdamW     | 7.54E+00   | 6.55E-02 | 44.98%    |
|                              | SLAKE      | 30    | 32         | 5.00E-05      | AdamW     | 7.03E-01   | 4.56E-02 | 67.58%    |
|                              | SLAKE      | 100   | 64         | 5.00E-05      | AdamW     | 7.29E+01   | 2.79E-01 | 72.72%    |
| VGG-GPT2                     | VQA-RAD    | 1     | 16         | 5.00E-05      | AdamW     | 1.88E+01   | 3.01E-01 | 29.93%    |
| OpenAI's GPT-2 and VGGNet-11 | Path-VQA   | 6     | 16         | 5.00E-05      | AdamW     | 1.83E+01   | 1.08E+00 | 28.00%    |
| VGG-BioGPT                   | SLAKE      | 1     | 16         | 5.00E-05      | AdamW     | 1.83E+01   | 3.10E-01 | 27.24%    |
| ViT-GPT2                     | VQA-RAD    | 6     | 16         | 5.00E-05      | AdamW     | 2.34E+01   | 6.12E+00 | 29.19%    |
|                              | SLAKE      | 6     | 16         | 5.00E-05      | AdamW     | 2.45E+01   | 6.12E+00 | 29.19%    |
| Idefics2-8B                  | VQA-RAD    | 1     | 1          | 1.00E-04      | AdamW     | 0.18       | -        | 53.22%    |
|                              | Path-VQA   | 1     | 1          | 1.00E-04      | AdamW     | 0.2        | -        | 31.50%    |
|                              | SLAKE      | 1     | 1          | 1.00E-04      | AdamW     | 0.04       | -        | 76.26%    |
| Idefics2-8B to ROCO          | VQA-RAD    | 3     | 1          | 1.00E-04      | AdamW     | 0.04       | -        | 76.26%    |
|                              | SLAKE      | 3     | 1          | 1.00E-04      | AdamW     | 0.1512     | -        | 81.00%    |



## File Description
1. vgg19gpt2.ipnyb is fine-tune file for ..............
2. blip.ipnyb ..............
3. idefics2.ipnyb ....................

## Demo
for demo vqa-task, visit our huggingface space : https://huggingface.co/spaces/KalbeDigitalLab/IDEFICS2-8B-MedicalVQA

## Reference :
- https://arxiv.org/pdf/2405.02246
- https://arxiv.org/pdf/2201.12086
- https://github.com/ab3llini/Transformer-VQA



