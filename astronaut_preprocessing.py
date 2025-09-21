import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace with your actual file path
df = pd.read_csv(r"c:\Users\harsh\OneDrive\Desktop\harshitaa_python_file\astronauts.csv")

print("=== INITIAL DATA EXPLORATION ===")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")

# Check for missing values
print(f"\n=== MISSING VALUES ANALYSIS ===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Check for duplicates
print(f"\n=== DUPLICATE ANALYSIS ===")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

def clean_astronaut_data(df):
    """
    Comprehensive preprocessing function for the astronaut database
    """
    df_clean = df.copy()
    
    print("\n=== STARTING DATA PREPROCESSING ===")
    
    # 1. HANDLE MISSING VALUES
    print("\n1. Handling Missing Values...")
    
    # Fill missing values based on column type and importance
    
    # Critical columns - keep NaN for analysis
    critical_cols = ['id', 'number', 'name', 'original_name']
    
    # Categorical columns - fill with mode or 'Unknown'
    categorical_fills = {
        'sex': 'Unknown',
        'nationality': 'Unknown', 
        'military_civilian': 'Unknown',
        'selection': 'Unknown',
        'occupation': 'Unknown'
    }
    
    for col, fill_value in categorical_fills.items():
        if col in df_clean.columns:
            mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else fill_value
            df_clean[col] = df_clean[col].fillna(mode_val)
    
    # Numerical columns - fill with median or 0 for mission-specific data
    numerical_cols = ['year_of_birth', 'year_of_selection', 'mission_number', 'total_number_of_missions',
                     'total_hrs_sum', 'field21', 'eva_hrs_sum', 'total_eva_hrs']
    
    for col in numerical_cols:
        if col in df_clean.columns:
            if 'eva' in col.lower() or 'hrs' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)  # Assume 0 hours if missing
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Mission-specific columns - fill with 'Unknown' or appropriate defaults
    mission_cols = ['mission_title', 'ascend_shuttle_orbit', 'descend_shuttle_orbit']
    for col in mission_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # 2. CLEAN AND STANDARDIZE TEXT DATA
    print("2. Cleaning and standardizing text data...")
    
    # Clean astronaut names
    if 'name' in df_clean.columns:
        df_clean['name'] = df_clean['name'].str.strip()
        df_clean['name_clean'] = df_clean['name'].str.replace(r'\s+', ' ', regex=True)
    
    if 'original_name' in df_clean.columns:
        df_clean['original_name'] = df_clean['original_name'].str.strip()
    
    # Standardize sex column
    if 'sex' in df_clean.columns:
        df_clean['sex'] = df_clean['sex'].str.lower().str.strip()
        sex_mapping = {'m': 'male', 'f': 'female', 'male': 'male', 'female': 'female'}
        df_clean['sex'] = df_clean['sex'].map(sex_mapping).fillna(df_clean['sex'])
    
    # Clean nationality
    if 'nationality' in df_clean.columns:
        df_clean['nationality'] = df_clean['nationality'].str.strip()
        # Standardize common nationality variations
        nationality_mapping = {
            'U.S.': 'USA',
            'U.S.S.R/Russia': 'Russia', 
            'U.S.S.R/Ru': 'Russia',
            'U.S.S.R': 'Russia'
        }
        df_clean['nationality'] = df_clean['nationality'].replace(nationality_mapping)
    
    # Clean military/civilian status
    if 'military_civilian' in df_clean.columns:
        df_clean['military_civilian'] = df_clean['military_civilian'].str.strip().str.lower()
    
    # Clean selection program
    if 'selection' in df_clean.columns:
        df_clean['selection'] = df_clean['selection'].str.strip()
    
    # Clean occupation
    if 'occupation' in df_clean.columns:
        df_clean['occupation'] = df_clean['occupation'].str.strip().str.lower()
    
    # Clean mission titles and shuttle names
    mission_text_cols = ['mission_title', 'ascend_shuttle_orbit', 'descend_shuttle_orbit']
    for col in mission_text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
    
    # 3. FEATURE ENGINEERING
    print("3. Creating new features...")
    
    # Age-related features
    if 'year_of_birth' in df_clean.columns:
        current_year = datetime.now().year
        df_clean['current_age'] = current_year - df_clean['year_of_birth']
        df_clean['birth_decade'] = (df_clean['year_of_birth'] // 10) * 10
    
    if 'year_of_selection' in df_clean.columns:
        df_clean['selection_decade'] = (df_clean['year_of_selection'] // 10) * 10
        
        # Age at selection
        if 'year_of_birth' in df_clean.columns:
            df_clean['age_at_selection'] = df_clean['year_of_selection'] - df_clean['year_of_birth']
    
    # Career span (if we have mission year data)
    if 'year_of_selection' in df_clean.columns and 'mission_number' in df_clean.columns:
        # This would need mission year data to calculate properly
        pass
    
    # Binary features
    if 'sex' in df_clean.columns:
        df_clean['is_male'] = (df_clean['sex'] == 'male').astype(int)
        df_clean['is_female'] = (df_clean['sex'] == 'female').astype(int)
    
    if 'military_civilian' in df_clean.columns:
        df_clean['is_military'] = (df_clean['military_civilian'] == 'military').astype(int)
    
    # Mission experience features
    if 'total_number_of_missions' in df_clean.columns:
        df_clean['is_veteran'] = (df_clean['total_number_of_missions'] > 1).astype(int)
        df_clean['mission_experience_level'] = pd.cut(df_clean['total_number_of_missions'], 
                                                    bins=[0, 1, 2, 5, float('inf')], 
                                                    labels=['Single', 'Couple', 'Multiple', 'Veteran'])
    
    # EVA experience
    eva_cols = ['eva_hrs_sum', 'total_eva_hrs']
    for col in eva_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_has_eva'] = (df_clean[col] > 0).astype(int)
            df_clean[f'{col}_category'] = pd.cut(df_clean[col], 
                                               bins=[-1, 0, 10, 50, float('inf')], 
                                               labels=['No_EVA', 'Short', 'Medium', 'Long'])
    
    # Total flight hours categories
    if 'total_hrs_sum' in df_clean.columns:
        df_clean['flight_hours_category'] = pd.cut(df_clean['total_hrs_sum'], 
                                                 bins=[0, 100, 500, 1000, float('inf')], 
                                                 labels=['Short', 'Medium', 'Long', 'Very_Long'])
    
    # Nationality groupings
    if 'nationality' in df_clean.columns:
        major_nations = ['USA', 'Russia', 'Japan', 'Germany', 'France', 'Italy', 'Canada']
        df_clean['nationality_group'] = df_clean['nationality'].apply(
            lambda x: x if x in major_nations else 'Other'
        )
    
    # 4. HANDLE OUTLIERS
    print("4. Handling outliers...")
    
    numerical_columns = ['year_of_birth', 'year_of_selection', 'total_number_of_missions', 
                        'total_hrs_sum', 'eva_hrs_sum', 'total_eva_hrs']
    
    outlier_info = {}
    for col in numerical_columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_info[col] = outlier_count
            
            # Flag outliers instead of removing them (important for astronaut data)
            df_clean[f'{col}_is_outlier'] = outliers_mask.astype(int)
    
    print(f"Outliers detected: {outlier_info}")
    
    # 5. REMOVE DUPLICATES
    print("5. Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    print(f"Removed {removed_duplicates} duplicate rows")
    
    # 6. DATA VALIDATION
    print("6. Performing data validation...")
    
    # Validate year ranges
    if 'year_of_birth' in df_clean.columns:
        invalid_birth = ((df_clean['year_of_birth'] < 1900) | 
                        (df_clean['year_of_birth'] > 2010)).sum()
        print(f"Invalid birth years: {invalid_birth}")
    
    if 'year_of_selection' in df_clean.columns:
        invalid_selection = ((df_clean['year_of_selection'] < 1950) | 
                           (df_clean['year_of_selection'] > 2025)).sum()
        print(f"Invalid selection years: {invalid_selection}")
    
    # Validate mission numbers
    if 'total_number_of_missions' in df_clean.columns:
        negative_missions = (df_clean['total_number_of_missions'] < 0).sum()
        print(f"Negative mission counts: {negative_missions}")
    
    return df_clean, outlier_info

def create_encoded_features(df_clean):
    """
    Create encoded versions of categorical variables
    """
    df_encoded = df_clean.copy()
    
    print("\n=== ENCODING CATEGORICAL VARIABLES ===")
    
    # Label encoding for ordinal/categorical variables
    label_encoders = {}
    
    categorical_cols = ['nationality', 'military_civilian', 'occupation', 'selection']
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle NaN values
            mask = df_encoded[col].notna()
            df_encoded.loc[mask, f'{col}_encoded'] = le.fit_transform(df_encoded.loc[mask, col].astype(str))
            df_encoded[f'{col}_encoded'] = df_encoded[f'{col}_encoded'].fillna(-1)
            label_encoders[col] = le
    
    # One-hot encoding for nominal variables with many categories
    onehot_cols = ['nationality_group', 'mission_experience_level', 'flight_hours_category']
    
    for col in onehot_cols:
        if col in df_encoded.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    return df_encoded, label_encoders

def scale_numerical_features(df_encoded):
    """
    Scale numerical features
    """
    df_scaled = df_encoded.copy()
    
    print("\n=== SCALING NUMERICAL FEATURES ===")
    
    # Select numerical columns for scaling
    numerical_cols = ['year_of_birth', 'year_of_selection', 'total_number_of_missions',
                     'total_hrs_sum', 'eva_hrs_sum', 'total_eva_hrs', 'current_age', 'age_at_selection']
    
    scaler = StandardScaler()
    
    cols_to_scale = [col for col in numerical_cols if col in df_scaled.columns]
    
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        print(f"Scaled columns: {cols_to_scale}")
    
    return df_scaled, scaler

def generate_preprocessing_report(df_original, df_clean):
    """
    Generate a comprehensive preprocessing report
    """
    print("\n" + "="*50)
    print("           PREPROCESSING REPORT")
    print("="*50)
    
    print(f"Original dataset shape: {df_original.shape}")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Rows removed: {df_original.shape[0] - df_clean.shape[0]}")
    print(f"New columns created: {df_clean.shape[1] - df_original.shape[1]}")
    
    print(f"\n=== MISSING VALUES SUMMARY ===")
    original_missing = df_original.isnull().sum().sum()
    final_missing = df_clean.isnull().sum().sum()
    print(f"Original missing values: {original_missing}")
    print(f"Final missing values: {final_missing}")
    print(f"Missing values handled: {original_missing - final_missing}")
    
    print(f"\n=== DATA TYPES SUMMARY ===")
    print("Final data types:")
    print(df_clean.dtypes.value_counts())
    
    print(f"\n=== KEY STATISTICS ===")
    if 'nationality' in df_clean.columns:
        print(f"Unique nationalities: {df_clean['nationality'].nunique()}")
        print(f"Top 5 nationalities:\n{df_clean['nationality'].value_counts().head()}")
    
    if 'sex' in df_clean.columns:
        print(f"\nGender distribution:\n{df_clean['sex'].value_counts()}")
    
    if 'total_number_of_missions' in df_clean.columns:
        print(f"\nMission statistics:")
        print(f"Average missions per astronaut: {df_clean['total_number_of_missions'].mean():.2f}")
        print(f"Max missions: {df_clean['total_number_of_missions'].max()}")

# Main execution
if __name__ == "__main__":
    print("Starting Astronaut Database Preprocessing...")
    
    # Store original dataset
    df_original = df.copy()
    
    # Apply preprocessing steps
    print("Step 1: Data Cleaning...")
    df_preprocessed, outlier_info = clean_astronaut_data(df)
    
    print("Step 2: Feature Encoding...")
    df_encoded, encoders = create_encoded_features(df_preprocessed)
    
    print("Step 3: Feature Scaling...")
    df_final, scaler = scale_numerical_features(df_encoded)
    
    # Generate comprehensive report
    generate_preprocessing_report(df_original, df_preprocessed)
    
    # Save processed datasets
    df_preprocessed.to_csv('astronaut_database_preprocessed.csv', index=False)
    df_encoded.to_csv('astronaut_database_encoded.csv', index=False)
    df_final.to_csv('astronaut_database_final_scaled.csv', index=False)
    
    print(f"\n" + "="*50)
    print("         PREPROCESSING COMPLETE!")
    print("="*50)
    print("Files saved:")
    print("1. astronaut_database_preprocessed.csv - Cleaned data")
    print("2. astronaut_database_encoded.csv - With encoded features") 
    print("3. astronaut_database_final_scaled.csv - Scaled features")
    
    print(f"\nPreprocessed dataset shape: {df_preprocessed.shape}")
    print(f"Final dataset columns: {df_final.shape[1]}")
    
    # Display sample of cleaned data
    print(f"\n=== SAMPLE OF CLEANED DATA ===")
    print(df_preprocessed[['name', 'sex', 'nationality', 'year_of_birth', 'total_number_of_missions', 
                          'total_hrs_sum', 'is_military', 'current_age']].head(10))
