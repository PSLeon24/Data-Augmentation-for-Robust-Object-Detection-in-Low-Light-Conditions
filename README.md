# Data Augmentation for Robust Object Detection in Low-Light Conditions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and experimental results for the research paper **"Low-Light Data Augmentation-based Object Detection for Nighttime Smart Yard Safety Management"**. The project introduces and validates a novel framework for enhancing object detection performance in the low-light and nighttime conditions frequently encountered in Smart Yard environments.

## 1. Abstract

In 24/7 Smart Yard environments, preventing safety accidents in low-light conditions such as nighttime is of paramount importance. However, object detection models used in conventional intelligent safety monitoring systems are predominantly trained on daylight data, leading to a significant degradation in detection performance for key safety objects like helmets, workers, and heavy equipment at night. This issue is exacerbated by the difficulty of acquiring sufficient low-light data from actual industrial sites. To address this problem, we propose a dual approach that combines data augmentation at the training stage with input data optimization at the inference stage. In the training stage, data generated using a diffusion model secures the model's generalization performance, while in the inference stage, real-time low-light enhancement pre-processing maximizes detection performance. In experiments conducted on the YOLOv11 model, the proposed method showed a 13%p improvement in overall object detection performance of mAP@50, from 0.35 to 0.48, compared to the baseline model trained only on the original dataset.

## 2. Research Goals

- **Build High-Fidelity Low-Light Datasets:** Generate realistic nighttime images from a standard daylight dataset using the InstructPix2Pix model, creating two distinct scenarios ('normal' and 'extreme' low-light) to ensure data diversity while preserving semantic consistency.
- **Enhance Model Generalization:** Train a YOLOv11 object detector on the augmented dataset to fundamentally improve its robustness and ability to recognize objects in various low-light conditions.
- **Optimize Inference Performance:** Design and select an optimal, lightweight pre-processing module (Gamma Correction) that can be applied in real-time during inference to enhance image quality with minimal computational overhead.
- **Develop and Validate a Hybrid Framework:** Systematically evaluate and prove the synergistic effect of combining generative data augmentation at the training stage with a real-time enhancement module at the inference stage, demonstrating superior performance over individual approaches.

## 3. Methodology

Our research pipeline is structured into four main phases:

### Phase 1: Data Augmentation

We use the [AI Hub Smart Yard Safety Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71770) as our base dataset. A generative model is used to create a low-light version for training.

- **Tool:** InstructPix2Pix, a conditional diffusion model that allows for precise, instruction-based image editing.
- **Process:** We convert daytime images into realistic nighttime scenes using specific text prompts designed to simulate 'normal' and 'extreme' low-light conditions.
- **Output Dataset:** The augmented low-light images are combined with the full original dataset to create the final training set, comprising 149,980 images.

### Phase 2: Model Training
<img width="1251" height="624" alt="image" src="https://github.com/user-attachments/assets/f818092d-a090-488f-9302-bcd80e23f041" />

- **Object Detector:** We use the **YOLOv11** architecture as the backbone for our object detection model.
- **Training:** A single, robust detector (`YOLOv11_Augmented`) is trained on the combined dataset (original + augmented) created in Phase 1. The goal is to produce a model with strong generalization performance across different lighting conditions.

### Phase 3: Pre-processing Module Selection

We design a two-stage evaluation process to select the optimal real-time pre-processing technique for the inference pipeline.

- **Candidates:** Traditional, computationally efficient methods: Histogram Equalization and Gamma Correction.
- **Stage 1 (Visual Quality):** Candidates are evaluated on their ability to restore a low-light image to its daytime counterpart. Performance is measured using **PSNR** and **SSIM** metrics.
- **Stage 2 (Detection Performance):** The best candidates from Stage 1 are applied during inference with the trained `YOLOv11_Augmented` model. The final selection is based on which technique yields the highest object detection performance (**mAP**).
- **Selected Module:** **Gamma Correction ($\gamma=0.5$)** was chosen as it demonstrated the best performance in both visual quality and final detection metrics.

### Phase 4: Evaluation

We conduct a comprehensive evaluation using a held-out test set to compare the performance of our proposed hybrid framework against several competing scenarios.

- **Key Scenarios:**
    1.  **Baseline:** `YOLOv11_Base` (trained on original data only) on low-light images.
    2.  **Pre-processing Only:** `YOLOv11_Base` + Gamma Correction on low-light images.
    3.  **Augmentation Only:** `YOLOv11_Augmented` on low-light images.
    4.  **Proposed Method (Hybrid):** `YOLOv11_Augmented` + Gamma Correction on low-light images.
- **Metrics:**
    - **Detection:** mAP@50, mAP@50-95, Precision, and Recall.
    - **Visual Quality:** PSNR, SSIM.

<img width="1340" height="1057" alt="image" src="https://github.com/user-attachments/assets/252daefe-84d6-416c-830a-2c97aac8ea27" />

<img width="1127" height="887" alt="image" src="https://github.com/user-attachments/assets/374964a1-dd4b-4caf-893f-05fc94a1e3ed" />


## 4. Repository Structure

```
.
├── SmartYard/                                   # Phase 1 – low-light data augmentation (InstructPix2Pix)
│   ├── instruct_pix2pix.py                      # first InstructPix2Pix test (timbrooks/instruct-pix2pix)
│   ├── instruct_pix2pix_Low-Light.py            # single-image day→night conversion
│   ├── instruct_pix2pix_Low-Light_experiment.py # prompt / parameter experiments
│   ├── instruct_pix2pix_Low-Light_experiment_multi.py  # multi-GPU batch generation of 'light'(l) / 'heavy'(h) low-light images
│   │                                            #   (num_inference_steps=15, image_guidance_scale=1.5, guidance_scale=7.5)
│   └── analyze.py                               # pixel-difference map between two images
└── YOLOv11/                                     # Phases 2–4 – training, pre-processing selection, evaluation
    ├── data_preprocess.ipynb, data_transform.ipynb   # AI Hub annotation → YOLO format, dataset re-organisation
    ├── data/test.ipynb, data/Validation_total/total.ipynb  # dataset checks / merged validation split
    ├── dataset.yaml, class.md                   # 5 classes: 낙하작업자, 연기, 조선작업자, 추락작업자, 충돌작업자
    ├── train.py, train-instruct-only.py         # YOLOv11 training (path pre-flight check + Ultralytics train);
    │                                            #   the two versions differ only in run name / GPU ids
    ├── validation.py                            # evaluate a trained model (model.val)
    ├── validation_{daylight,lowlight,total}_{gamma,histogram}.py  # Phase 3/4: evaluation with Gamma / HistEq pre-processing
    ├── validation_gamma.py, validation_histogram.py, validation_daylight.py
    ├── validate_*.yaml                          # evaluation splits (daylight / lowlight / total × pre-processing)
    ├── inference.py                             # inference demo
    ├── args.yaml, results.csv                   # training config & per-epoch metrics of the released model
    └── yolo11n.pt                               # Ultralytics YOLO11n pretrained weight (initialisation)
```

### Reproduction workflow

1. **Data** – download the [AI Hub Smart Yard Safety Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71770) (not redistributed here) and convert it with `YOLOv11/data_preprocess.ipynb`.
2. **Augmentation** – generate low-light images with `SmartYard/instruct_pix2pix_Low-Light_experiment_multi.py` and merge them with the original training set (`data_transform.ipynb`).
3. **Training** – `python YOLOv11/train.py` with `dataset.yaml` pointing to the original or original+augmented training set (YOLO11n, 100 epochs; see `args.yaml`).
4. **Pre-processing selection & evaluation** – run `validation_*_gamma.py` / `validation_*_histogram.py` with the matching `validate_*.yaml` to compare Gamma (γ = 0.5) and Histogram Equalization on daylight / low-light / total splits.

> Note: dataset paths in the YAML files and scripts are absolute paths from the original training server (`/SSD4/psleon/YOLOv11/...`) and must be adapted.
> Code added on 2026-10-03 from the original experiment server for reproducibility (no functional changes).
