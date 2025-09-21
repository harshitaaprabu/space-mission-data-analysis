# Astronaut Database Preprocessing

A comprehensive data preprocessing pipeline for NASA/Roscosmos astronaut mission database.

## Overview
This project processes astronaut mission data from NASA and Roscosmos, cleaning and preparing it for analysis. The dataset contains information about astronauts' backgrounds, missions, EVA activities, and career statistics.

## Features
- ✅ Data cleaning and standardization
- ✅ Missing value handling
- ✅ Feature engineering (age calculations, experience levels)
- ✅ Categorical encoding (Label Encoding, One-Hot Encoding)
- ✅ Outlier detection and flagging
- ✅ Data validation and quality checks
- ✅ Multiple output formats for different use cases

## Dataset Information
- **Source**: NASA, Roscosmos, and fan-made websites
- **Coverage**: All astronauts who participated in space missions before January 15, 2020
- **Records**: ~500+ astronaut mission records
- **Features**: Name, nationality, birth year, selection program, mission details, EVA hours

## Installation

### Requirements
- Python 3.7+
- Required packages (see requirements.txt)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/astronaut-data-preprocessing.git
cd astronaut-data-preprocessing

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python astronaut_preprocessing.py
