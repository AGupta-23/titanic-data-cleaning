# Titanic Data Cleaning Project

## Overview
This project performs data cleaning and preprocessing on the famous Titanic dataset. The script handles missing values, data types, and prepares the dataset for further analysis or machine learning tasks.

## Dataset
- **Source**: Titanic passenger data
- **Original Shape**: 891 rows, 12 columns
- **Final Shape**: 891 rows, 10 columns (after cleaning)

## Data Cleaning Steps

The cleaning process includes:

1. **Missing Value Treatment**:
   - Age: Filled with median value
   - Embarked: Filled with mode (most frequent value)
   - Fare: Filled with median value

2. **Feature Engineering**:
   - Removed unnecessary columns
   - Handled categorical variables
   - Standardized data types

## Files Structure
```
titanic-data-cleaning/
├── data/           # Input dataset
├── src/           # Source code
│   └── clean_titanic.py
└── output/        # Cleaned dataset output
```

## How to Run

1. Make sure you have Python and pandas installed:
```bash
pip install pandas
```

2. Run the cleaning script:
```bash
python src/clean_titanic.py
```

## Output
![Screenshot 2025-05-26 204817](https://github.com/user-attachments/assets/1f9cff40-beb4-4f54-8630-3fae8d71c31e)
The script outputs:
- Cleaned dataset in the `output/` folder
- Data summary and statistics
- Information about the cleaning process

## Requirements
- Python 3.x
- pandas
- numpy 
