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
The table below shows the evaluation results for the fine-tuned models on the medical datasets.

## Evaluation
The table below shows the accuracy results for the pre-fine-tuned models on the VQA-RAD and SLAKE datasets.

| Model        | Pre-fine-tune Dataset | VQA-RAD Accuracy | SLAKE Accuracy |
|--------------|-----------------------|------------------|----------------|
| BLIP         | -                     | 44.57%           | 67.58%         |
| VGG-GPT2     | -                     | 27.72%           | 28.09%         |
| VGG-BioGPT   | -                     | 27.40%           | 27.24%         |
| Idefics2-8B  | -                     | 53.22%           | 76.26%         |
| Idefics2-8B  | ROCO                  | 51.00%           | 81.00%         |


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



