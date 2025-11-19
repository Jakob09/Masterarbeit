# Effect of Adversarial Attacks on Saliency Map Explanations of Deep Residual Neural Networks 🧠

## 📋 Table of Contents

1.  [About the Project](#1-about-the-project)
2.  [Prerequisites](#2-prerequisites)
3.  [Setup and Installation](#3-setup-and-installation)
4.  [Usage](#4-usage)

---

## 1. About the Project

This repository investigates the impact of **adversarial attacks** on the explainability of Deep Residual Neural Networks (ResNets), specifically focusing on **Saliency Map** explanations.

The project systematically generates adversarial examples using various methods, calculates the corresponding Saliency Maps for both the original and adversarial inputs, and quantifies the change in these explanations using a set of comparative metrics. The goal is to assess the robustness of Saliency Maps as a trustworthy explainability method in the presence of minor input perturbations.

---

## 2. Prerequisites

You need **Python 3.13** and a working environment manager (like `conda` or `venv`) to run the experiments.

---

## 3. Setup and Installation

### 3.1. Environment Setup

The necessary dependencies are listed in the `requirements.txt` file. 


If you use Conda, you can set up the required environment using the following commands:

1.  **Create a Conda environment** 
    ```bash
    conda create -n adv_saliency python=3.13
    ```

2.  **Activate the environment:**
    ```bash
    conda activate adv_saliency
    ```

3.  **Install dependencies** using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Deactivate the environment** when finished:
    ```bash
    conda deactivate
    ```


## 4. Usage

1. Pipeline Execution
The core logic is divided into two Python scripts handling the adversarial generation and metric calculation. The results are stored in CSV files.

1.1. Untargeted Attacks
The batch_wise_pipeline.py script generates untargeted adversarial examples for various attacks available in the Foolbox library. It computes saliency maps and comparison metrics for each image. The script takes a command-line argument indicating the index of the attack-group to use (all attacks are given in a list of dictionaries, containing the foolbox attack and the epsilon values to use for the adversarial attack).

1.2. Targeted Attacks
The targeted_batch_wise_pipeline.py script generates targeted adversarial examples for various attacks available in the Foolbox library. It computes saliency maps and comparison metrics for each image. The script takes a command-line argument indicating the index of the attack-group to use (all attacks are given in a list of dictionaries, containing the foolbox attack and the epsilon values to use for the adversarial attack).

2. Metric Analysis
The analysis of the calculated metrics is executed in various jupyter notebooks ranging from EDA to clustering of the methods/attacks.

3. Modifications

3.1. Other datasets
Different datasets can be used by editing the load_and_transform_images() function in the batch_wise_pipeline.py. The function should return a tensor of preprocessed images and a tensor of labels (class_ids)

3.2. Other models
Other models can be used by modifying the following code (using a pytorch model)

```python
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).eval()
model = model.to(device)

preprocess = weights.transforms()
bounds = (-2.4, 2.8)

fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)
```
