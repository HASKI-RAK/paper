# Analysis of Benchmarking Results

This repository contains the analysis of benchmarking results from our study. The provided code allows you to reproduce all computations and visualizations presented in the paper.

## Repository Structure
```
repository/
├── data/              # Contains the datasets used in the analysis
├── helpers/           # Utility functions to support the main analysis
├── src/               # Additional source code handling the datasets
├── visualization/     # Provides visualization functions

```

## Environment Setup

For reproducibility, you should setup a Virtual Python environment. Please follow the commands below to set up your environment and install all dependencies:

### Create and Activate Virtual Environment

- **Linux/macOS:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### Install Required Dependencies

Once the virtual environment is activated, install the necessary packages:
```bash
pip install -r requirements.txt
```

## Running the Analysis

The repository includes two paired files, `main.py` and `main.ipynb`, which contain identical code. Run `main.py` for a straightforward script-based execution, or choose the notebook version if you wish to generate an HTML report that includes additional documentation stored in the notebook. You can choose based on your preference:

### For Linux/macOS:

- **Using `main.py` (script execution):**
  ```bash
  MPLBACKEND=Agg python main.py
  ```

- **Using `main.ipynb` (notebook execution with HTML export):**
  ```bash
  MPLBACKEND=Agg jupyter nbconvert --execute --to html main.ipynb
  ```

### For Windows (PowerShell):

- **Using `main.py` (script execution):**
  ```powershell
  $env:MPLBACKEND="Agg"; python main.py
  ```

- **Using `main.ipynb` (notebook execution with HTML export):**
  ```powershell
  $env:MPLBACKEND="Agg"; jupyter nbconvert --execute --to html main.ipynb
  ```

### Why should I provide the `MPLBACKEND` environment variable?
 
This environment variable instructs Matplotlib to use a non-interactive backend (Agg), which prevents the execution from pausing when generating plots. That means `plt.show()` calls are effectively suppressed, which is fine since all plots are automatically saved in the directory `./exports`.


## Reproducibility

This repository is designed to ensure that every analysis can be fully reproduced. Whether you run the script or the notebook, the output and generated plots will remain consistent. If you encounter any issues or have questions, please feel free to contact the authors.

