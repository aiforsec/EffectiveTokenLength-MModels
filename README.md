# EffectiveTokenLength-MModels
**SIGIR'25 Submission- Resource and Reproducibility Paper**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## Overview

In modern multimodal architectures, models process diverse types of input data where tokenization plays a critical role in performance. However, the actual number of tokens that effectively contribute to the model's output—known as the *effective token length*—can be quite different from the maximum token capacity. This repository addresses that challenge by providing:

- **Analytical tools** to measure effective token lengths.
- **Benchmarking scripts** to evaluate tokenization strategies.
- **Visualization utilities** to explore token distributions and their impact on performance.

---

## Features

- **Multimodal Analysis:** Assess effective token lengths across text, image, and hybrid data.
- **Benchmarking Tools:** Run evaluations to see how token usage correlates with model performance.
- **Visualization:** Generate plots and reports to better understand token efficiency.
- **Extensible Architecture:** Easily integrate new experiments or modify existing ones.

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aiforsec/EffectiveTokenLength-MModels.git
   cd EffectiveTokenLength-MModels

2. **Create a virtual environment(recommended):**
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
    pip install -r requirements.txt

---

## Datasets

This project relies on four main datasets: **Urban1k**, **Roco**, **Factify**, and **ShareGPT4V**. Each dataset has different hosting and access requirements, so please follow the instructions below to download or request access.

---

### Urban1k

- **Description:** Urban1k is a general domain dataset
- **Download Instructions:**
  1. Download the zip file from [Urban1k](https://huggingface.co/datasets/BeichenZhang/Urban1k)
---

### Roco
 
- **Description:** Roco is a dataset for medical image captioning.
  1. Download from here [ROCO](https://www.kaggle.com/datasets/virajbagal/roco-dataset)
  2. Open test folder, open radiology folder, download the images folder and corresponding csv file.

---

### Factify

- **Description:** Factify is a fake news dataset
- **Access Instructions:**
  1. On opening the [Factify Github](https://github.com/surya1701/Factify-2.0) visit the codalabs link and request permission for the dataset.
  2. Once permission is granted, you will receive a download link or instructions on how to access the dataset.

---

### ShareGPT4V
- **Description:** ShareGPT4V is a general domain dataset.
- **Access Instructions:**
  1. Access the dataset from here [SHAREGPT4V](https://sharegpt4v.github.io/)
  2. Download 100k json file and image zips, then extract the images in separate folder

---

## Preprocessing

After obtaining each dataset, you need to preprocess them to ensure they match the format expected by the analysis pipeline.
Run the specific preprocessfile for the dataset before running benchmark. 

---

## Usage

### Running the Analysis

Before running the analysis, open the `config.yaml` file and adjust any parameters necessary for your specific setup—such as dataset paths, model names, and output directories. Once your configuration is set, you can execute the analysis with:

   ```bash
   python benchmark.py
   ```
## Contact 

If you have any questions or require further assistance, please reach out to us at ln8378@rit.edu
