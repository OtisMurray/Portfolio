# PM2.5 Air Pollution Data Processing Project

This project processes large-scale global PM2.5 air pollution data from NetCDF (.nc) files and extracts a cleaned dataset focused on North America. The output dataset can be used for analysis, visualization, and predictive modeling of air pollution and its impacts.

---

## Project Structure

project/
│── data/
│   ├── raw/        
│   └── processed/
│
│── src/
│   ├── extract_pm25.py
│   ├── model.py
│   └── other scripts
│
│── requirements.txt
│── README.md

---

## Setup

Install all required libraries:

pip install -r requirements.txt

---

## Data

Download the dataset from:
[INSERT ZENODO LINK HERE]

After downloading, place all .nc files into:

data/raw/

---

## How to Run

To generate the cleaned PM2.5 dataset:

python src/extract_pm25.py

---

## Output

The script will generate a cleaned dataset in:

data/processed/na_pm25_cells_clean.csv

The dataset includes:
- Latitude
- Longitude
- PM2.5 concentration
- Date

---

## Notes

- Data is downsampled for efficiency
- Only North America is included
- Missing PM2.5 values are removed
- Ensure all required .nc files are placed in data/raw/ before running

---

## Reproducibility

All dependencies required to run this project are listed in requirements.txt. Running the setup and execution steps above will reproduce the results.

---

## Authors

Alfred Liljas
The Pennsylvania State University


Otis Murray  
The Pennsylvania State University
Major in Statistical Modeling Data Science, Minor in Math