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

## Installation

Clone the repository:

```bash
git clone https://github.com/belinskymatus/-Disaggregation-of-household-electricity-consumption.git
cd -Disaggregation-of-household-electricity-consumption
(Optional) Create and activate a virtual environment:

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
Install the required Python packages (see section above).

Running the notebooks
Start Jupyter Lab or Jupyter Notebook:

jupyter lab
# or
jupyter notebook
Alternatively, open the project in Visual Studio Code and use the built-in Jupyter support.file:1

Open the desired notebook in the browser:

Downsampling.ipynb

Kmeansmodel.ipynb

FHMMmodel.ipynb

Seq2seq.ipynb

Execute the notebook cells sequentially (typically with Shift + Enter).file:1

Working with data
All input files are stored in the data folder.file:1

Downsampling.ipynb
This notebook:

loads original CSV files (e.g., redd_house1_0.csv to redd_house1_6.csv)

removes unnecessary columns

creates a simulated time axis

performs resampling to 5-second intervals using averaging

saves the result as standardized_house1_X.csv, which is used as input for the models.file:2

If you change the location or names of data files, update the file paths in the notebooks (pd.read_csv, open, read_json, etc.):

df = pd.read_csv("data/standardized_house1_0.csv")
Model overview
Combinatorial Optimization – Kmeansmodel.ipynb
This notebook implements a heuristic Combinatorial Optimization (CO) method using K-means clustering on combined training data from standardized_house1_0.csv to standardized_house1_5.csv.file:2

Key steps:file:2

build a training matrix from multiple CSV files

train the CO model on typical appliance consumption patterns

disaggregate a test dataset (standardized_house1_6.csv) and optionally real data in data.json

compute MAE, MSE, RMSE

visualize aggregate and disaggregated signals, including random time windows

Factorial Hidden Markov Model – FHMMmodel.ipynb
This notebook implements disaggregation using an FHMM, with separate HMMs per appliance combined into a single model.file:2

Workflow:file:2

load prepared CSV files standardized_house1_0.csv to standardized_house1_5.csv

train the FHMM on aggregate + appliance-level data

disaggregate the test file standardized_house1_6.csv and data from data.json

evaluate MAE, MSE, RMSE

plot aggregate consumption in black and predicted appliance signals in different colors

Seq2Seq CNN model – Seq2seq.ipynb
This notebook contains a Seq2Seq model with convolutional layers, where the input is a sequence of aggregated consumption and the output is the predicted consumption of one appliance.file:1

Main ideas:file:2

define sequence length, batch size, number of epochs, and lists of training/test CSV files

create overlapping sequences (create_sequences), normalize and denormalize data, load input–target pairs per appliance

train a separate model for each appliance on data from standardized_house1_0.csv to standardized_house1_5.csv

test on standardized_house1_6.csv and data.json

compute MAE, MSE, RMSE and visualize hourly windows and full-series plots

Evaluation and visualization
For all models, standard error metrics are computed between actual and predicted consumption:file:1

MAE – Mean Absolute Error

MSE – Mean Squared Error

RMSE – Root Mean Squared Error

The notebooks also:

print actual vs. predicted consumption per appliance

generate plots comparing total consumption with disaggregated components

show zoomed-in time windows for detailed inspection.file:1

Author
Author: Matúš Belinskýfile:1
Faculty: Faculty of Electrical Engineering and Informatics, Technical University of Košicefile:1
Study program / field: Business Informatics / Informaticsfile:1
Year: 2025file:1

