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
The table below shows the accuracy results for the pre-fine-tuned models on the VQA-RAD and SLAKE datasets.

| Model        | Pre-fine-tune Dataset | VQA-RAD Accuracy | SLAKE Accuracy |
|--------------|-----------------------|------------------|----------------|
| BLIP         | -                     | 44.57%           | 67.58%         |
| VGG-GPT2     | -                     | 27.72%           | 28.09%         |
| VGG-BioGPT   | -                     | 27.40%           | 27.24%         |
| Idefics2-8B  | -                     | 53.22%           | 76.26%         |
| Idefics2-8B  | ROCO                  | 51.00%           | 81.00%         |

## File Description
in notebooks folder, there are files for fine-tuning model in this project
1. notebooks/BLIP(vqa_base)_SLAKE.ipynb
2. notebooks/BLIP(vqa_base)_VQARAD.ipynb
3. notebooks/FT_Idefics_2_ROCO.ipynb
4. notebooks/FT_Idefics_2_VQA.ipynb
5. notebooks/VGG_GPT2_BioGPT.ipynb
6. notebooks/preprocessing_PMCVQA_small.ipynb
7. notebooks/preprocessing_ROCO.ipynb
8. notebooks/preprocessing_SLAKE.ipynb

## Demo
for demo vqa-task, visit our huggingface space : https://huggingface.co/spaces/KalbeDigitalLab/IDEFICS2-8B-MedicalVQA
app.py file to run interface demo for idefics2

## Reference :
- https://arxiv.org/pdf/2405.02246
- https://arxiv.org/pdf/2201.12086
- https://github.com/ab3llini/Transformer-VQA

