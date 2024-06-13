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
The project addresses the challenge of accurate and efficient medical imaging analysis in healthcare, aiming to reduce human error and workload for radiologists. The proposed solution involves developing advanced AI models for Visual Question Answering (VQA) to assist healthcare professionals in analyzing medical images quickly and accurately. These models will be integrated into a user-friendly web application, providing a practical tool for real-world healthcare settings.

## Dataset
this project fine-tune pre-trained model using these datasets :
rad-vqa dataset : https://huggingface.co/datasets/flaviagiammarino/vqa-rad
slake dataset : https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english
roco dataset : https://huggingface.co/datasets/mdwiratathya/ROCO-radiology

Reference Paper: [Medical visual question answering: A survey](https://www.sciencedirect.com/science/article/abs/pii/S0933365723001252)

## Model Architecture
<!-- The model uses a Parameterized Hypercomplex Shared Encoder network (PHYSEnet). -->

![Model Architecture](img/Model-Architecture.png)

Reference: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0933365723001252)

## Evaluation

## File Description
## Demo
Please select the example below or upload 4 pairs of mammography exam results.

## Usage

```
Run the following command on below
Python app.py
```
## Huggingface
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
