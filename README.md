# DeepSolar-3M

**📄 Paper:** [DeepSolar-3M: An AI-Enabled Solar PV Database Documenting 3 Million Systems Across the US](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2025/55/paper.pdf)  
**📍 Conference:** [ICLR 2025 - Tackling Climate Change with Machine Learning Workshop](https://www.climatechange.ai/papers/iclr2025/55)

---

## Overview

DeepSolar-3M provides fast, high-resolution mapping of rooftop photovoltaic (PV) systems across the United States.  
This repository contains county-level and blockgroup-level aggregated data from our AI pipeline.

**Key features:**
- Scalable detection of PV installations from aerial imagery 
- Blockgroup-level and county-level aggregation of PV system statistics
- Detailed breakdowns by system type (residential, commercial, utility-scale, solar heating)

---

## 📊 County-Level Dataset

Each entry corresponds to a U.S. county (identified by FIPS code) and includes:

- **Total PV system count**  
- **Total PV area** (in square meters)  
- **Median PV area** (m²)  
- **Average PV area** (m²)

**Breakdown by system type (% of systems):**
- Residential systems
- Commercial systems
- Utility-scale systems
- Solar heating systems

---

## 🗺️ Block Group-Level Dataset

Each entry corresponds to a U.S. Census block group (identified by GEOID/Block Group FIPS) and includes all the features listed above.

---

## 🛰️ Solar Models (`solar_models/`)

This folder contains code and utilities for running the DeepSolar-3M segmentation and detection models on new imagery.

**Subfolders:**
- `src/` — Inference notebook and dataloader script.
- `data_folder/` — Example input images for testing.

### Downloading Model Weights

You can download the required model files from our [Google Drive link](https://drive.google.com/drive/folders/1f7HNsncANmPvy1BfMoazgbaIs1b8TVFc?usp=drive_link).

---

## 📬 Citation

If you find this resource useful, please cite:

```
@article{PRABHA2026127411,
title = {Nationwide insights on solar deployment trends and spatial inequalities revealed by vision transformer models},
journal = {Applied Energy},
volume = {408},
pages = {127411},
year = {2026},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2026.127411},
url = {https://www.sciencedirect.com/science/article/pii/S0306261926000632},
author = {Rajanie Prabha and Zhecheng Wang and Chad Zanocco and June Flora and Ram Rajagopal},
keywords = {Solar energy, Solar PV adoption, Deep learning, Vision transformers, Lorenz curve, Spatial inequality, Energy policy},
}

@inproceedings{prabha2025deepsolar3m,
  title={DeepSolar-3M: An AI-Enabled Solar PV Database Documenting 3 Million Systems Across the US},
  author={Prabha, Rajanie and Wang, Zhecheng and Zanocco, Chad and Flora, June and Rajagopal, Ram },
  booktitle={ICLR 2025 Workshop on Tackling Climate Change with Machine Learning},
  url={https://www.climatechange.ai/papers/iclr2025/55},
  year={2025}
}

```
---


---

## Contact

Feel free to reach out in case you have any questions - ```rajanie@stanford.edu``` 

---
