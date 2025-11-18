# Effect of Adversarial Attacks on Saliency Map Explanations of Deep Residual Neural Networks 🧠

## 📋 Table of Contents

1.  [About the Project](#1-about-the-project)
2.  [Prerequisites](#2-prerequisites)
3.  [Setup and Installation](#3-setup-and-installation)
4.  [Data](#4-data)
5.  [Pipeline Execution](#5-pipeline-execution)
6.  [Analysis and Results](#6-analysis-and-results)
7.  [Contributing](#7-contributing)
8.  [License](#8-license)

---

## 1. About the Project

This repository investigates the impact of **adversarial attacks** on the explainability of Deep Residual Neural Networks (ResNets), specifically focusing on **Saliency Map** explanations.

The project systematically generates adversarial examples using various methods, calculates the corresponding Saliency Maps for both the original and adversarial inputs, and quantifies the change in these explanations using a set of comparative metrics. The goal is to assess the robustness of Saliency Maps as a trustworthy explainability method in the presence of minor input perturbations.

---

## 2. Prerequisites

You need **Python 3.x** and a working environment manager (like `conda` or `venv`) to run the experiments.

---

## 3. Setup and Installation

### 3.1. Environment Setup

The necessary dependencies are listed in the `requirements.txt` file. You can set up the environment and install all packages using the following commands:

Using `pip` and `venv`:
```bash
# Create a virtual environment
python -m venv venv
# Activate the environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt


## 4. Data
