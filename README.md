# Disaggregation of Household Electricity Consumption

This repository contains the source code and Jupyter notebooks for a bachelor thesis focused on disaggregating household electricity consumption into individual appliance usage based on aggregated meter data.[file:1][file:2]  
The application enables training and comparing several models (Combinatorial Optimization, FHMM, Seq2Seq CNN) using standard error metrics (MAE, MSE, RMSE) and clear visualizations.[file:1][file:2]

---

## Repository structure

The repository primarily contains the following components:[file:1][file:2]

- `Downsampling.ipynb` – notebook for decreasing the time resolution of input data (resampling from 1 s to 5 s) and standardizing them into `standardized_house1_X.csv` format.[file:2]  
- `Kmeansmodel.ipynb` – implementation of a heuristic Combinatorial Optimization (CO) model using K-means clustering, including training, testing, metric calculation, and visualization (slices and statistics).[file:1][file:2]  
- `FHMMmodel.ipynb` – implementation of a Factorial Hidden Markov Model (FHMM) using the `hmmlearn` library, including training, disaggregation, metric calculation, and plots.[file:1][file:2]  
- `Seq2seq.ipynb` – implementation of a deep Seq2Seq model based on convolutional neural networks (CNN) for disaggregation of individual appliance consumption.[file:1][file:2]  
- `data/` – directory with input data:  
  - CSV files `standardized_house1_0.csv` to `standardized_house1_6.csv`,  
  - JSON file `data.json` containing real aggregated measurements.[file:1][file:2]  
- Auxiliary files (e.g., `fhmm_model.py` with the FHMM implementation, if included in the project as described in the system manual).[file:2]

---

## Application purpose

The goal of the application is to estimate the consumption of individual electrical appliances in a household over time based on a single measurement of total aggregated power consumption.[file:1][file:2]  
The user can select a specific model, train it on available data, run disaggregation on test or custom data, and then analyze the results using error metrics and visualizations.[file:1][file:2]

---

## System requirements

**Supported operating systems**[file:1][file:2]

- Windows 10 / 11 (64-bit)  
- Ubuntu Linux 18.04+  
- macOS 10.15 (Catalina) or newer  

For the Seq2Seq model with GPU acceleration, a system with an NVIDIA GPU supporting CUDA (Compute Capability ≥ 3.5) is recommended.[file:1][file:2]

**Software requirements**[file:1][file:2]

- Python 3.8 – 3.11  
- Jupyter Lab or Visual Studio Code (with Jupyter notebook support)

**Hardware requirements (minimum / recommended)**[file:1][file:2]

- CPU: 64‑bit, min. 2 cores / recommended 4+ cores  
- RAM: min. 4 GB / recommended 8+ GB  
- GPU: optional / recommended NVIDIA GPU with CUDA for faster Seq2Seq training  
- Storage: min. 250 MB free space / recommended 500 MB

---

## Python dependencies

To run the application, install the following Python libraries (versions based on the manuals):[file:1][file:2]

- TensorFlow 2.16.1  
- Keras 3.1.1  
- scikit-learn 1.3.2  
- matplotlib 3.8.3  
- pandas 2.2.2  
- hmmlearn 0.3.0  
- seaborn 0.13.2  
- numpy 1.24.4  

You can install everything at once with:[file:1]

```bash
pip install tensorflow keras scikit-learn matplotlib pandas hmmlearn seaborn numpy
