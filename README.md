# Data Augmentation for Robust Object Detection in Adverse Weather Conditions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and experimental results for the research project investigating and comparing data augmentation techniques for robust object detection in adverse weather conditions, specifically within a Smart Yard environment.

## 1. Abstract

Vision-based systems are integral to modern Smart Yard operations, but their performance degrades significantly under adverse weather conditions such as snow, rain, and fog. This project aims to enhance the robustness of object detection models (YOLO & DETR) by leveraging generative AI for data augmentation. We systematically compare two state-of-the-art diffusion-based image editing techniques: **ControlNet** and **Instruct Pix2Pix**. Furthermore, we evaluate the effectiveness of this data augmentation strategy against a traditional approach of using a dedicated image restoration pre-processing module. The study utilizes the "Smart Yard Environment Object Detection Dataset" provided by AI Hub.

## 2. Research Goals

- **Build High-Fidelity Weather Datasets:** Generate realistic snow, haze, and rain effects on the clean Smart Yard dataset using both ControlNet and Instruct Pix2Pix, while preserving the original content fidelity.
- **Compare Augmentation Techniques:** Quantitatively and qualitatively evaluate whether ControlNet (geometric control) or Instruct Pix2Pix (semantic control) is more effective for training robust object detectors.
- **Evaluate Competing Strategies:** Compare the performance of two distinct strategies:
    1.  **Robust Model Strategy:** Training a detector on an augmented dataset.
    2.  **Restoration Strategy:** Using an image restoration module to clean an image before feeding it to a detector trained on clean data.
- **Provide Comprehensive Analysis:** Analyze the performance of different object detection architectures (YOLO vs. DETR) and their response to each enhancement strategy.

## 3. Methodology

Our research pipeline is divided into four main phases:

### Phase 1: Data Augmentation

We use the [AI Hub Smart Yard Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71770) as our Ground Truth (GT). Two parallel tracks are used to generate augmented training data:

-   Track A: ControlNet (`Dataset_CN`)
    -   We extract structural maps (e.g., Canny Edges, Depth) from GT images.
    -   These maps are used as explicit conditions in ControlNet to guide the diffusion process, ensuring geometric fidelity while adding weather effects via text prompts (e.g., `photorealistic heavy snow`).

-   Track B: Instruct Pix2Pix (`Dataset_IP2P`)
    -   We use the original GT image and a text instruction as direct input.
    -   Editing instructions (e.g., `"add dense fog to the scene"`) guide the model to perform semantic image editing.

### Phase 2: Model Training

-   Object Detectors:
    -   We train two architectures: **YOLOv11** and **DETR**.
    -   For each architecture, three models are trained:
        1.  `Detector_Base`: Trained only on the original GT dataset.
        2.  `Detector_CN`: Trained on GT + `Dataset_CN`.
        3.  `Detector_IP2P`: Trained on GT + `Dataset_IP2P`.

-   Image Restoration Module (`M_restore`):
    -   A state-of-the-art restoration model (e.g., NAFNet) is trained on pairs of `(adverse_weather_image, GT_image)` created in Phase 1.

### Phase 3: Evaluation

We conduct a comprehensive evaluation using a held-out test set to compare the performance of different pipelines under various weather conditions.

-   **Key Scenarios:**
    1.  **Baseline:** `Detector_Base` on weather images.
    2.  **Augmentation (ControlNet):** `Detector_CN` on weather images.
    3.  **Augmentation (IP2P):** `Detector_IP2P` on weather images.
    4.  **Restoration:** `M_restore` -> `Detector_Base` on weather images.
-   **Metrics:**
    -   **Detection:** mAP@0.5, mAP@0.5:0.95, Precision, Recall
    -   **Restoration:** PSNR, SSIM

## 4. Repository Structure
- TBD
