#!/usr/bin/env python
# coding: utf-8


import statistics
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


# Load dataframe
CO2_init = pd.read_csv(
    "data/raw/CO2_passenger_cars.csv", sep=";", low_memory=False)

# DATA PRE-PROCESSING #

# Only german registered vehicles to be considered
CO2_init_DE = CO2_init.loc[CO2_init['MS'] == 'DE']

# Feature Selection #

# Features to be deleted of dataframe
CO2_raw_raw = CO2_init_DE.drop(columns=['Cn', 'MS', 'ID', 'VFN', 'Mh', 'MMS', 'TAN', 'Man',
                                        'Va', 'Ve', 'Mt', 'Ewltp (g/km)', 'Ernedc (g/km)',
                                        'Erwltp (g/km)', 'De', 'Vf', 'r', 'Ct', 'Cr', 'It',
                                        'T', 'z (Wh/km)'])


# Final features to be deleted
CO2_raw = CO2_init_DE.drop(columns=['MS', 'ID', 'VFN', 'Mh', 'MMS', 'TAN', 'Man', 'Va', 'Ve',
                                    'Mt', 'Ewltp (g/km)', 'Ernedc (g/km)', 'Erwltp (g/km)',
                                    'De', 'Vf', 'r', 'At2 (mm)', 'Ct', 'Cr', 'It', 'T',
                                    'z (Wh/km)'])


# Fuel types homoganization

CO2_raw['Ft'] = CO2_raw['Ft'].replace(['DIESEL'], 'Diesel')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['PETROL'], 'Petrol')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['ELECTRIC', 'electric'], 'Electric')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['petrol'], 'Petrol')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['diesel'], 'Diesel')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['DIESEL                   '], 'Diesel')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['PETROL                   '], 'Petrol')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(
    ['Electric                 '], 'Electric')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['PETROL/ELECTRIC'], 'Petrol/Electric')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['LPG                      '], 'LPG')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(
    ['NG-biomethane            '], 'NG-biomethane')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['NG-BIOMETHANE'], 'NG-biomethane')
CO2_raw['Ft'] = CO2_raw['Ft'].replace(['PETROL/ELECTRIC'], 'Petrol/Electric')

# Drop the vehicle make
CO2_raw_final = CO2_raw.drop(columns=['Cn'])


# Pre-processing #

# A MISSING VALUES

# Imputing missing categorical values
imp_most_frequent = SimpleImputer(strategy='most_frequent')
for_imputing_cat = ['Mp', 'Mk']
for feature in for_imputing_cat:
    CO2_raw_final[feature] = imp_most_frequent.fit_transform(
        CO2_raw_final[feature].to_numpy().reshape(-1, 1))

# Imputing numerical values
for_imputing_num = ['m (kg)', 'Enedc (g/km)', 'W (mm)',
                    'At1 (mm)', 'ec (cm3)', 'ep (KW)']
for feature in for_imputing_num:
    CO2_raw_final[feature] = CO2_raw_final[feature].fillna(
        CO2_raw_final[feature].median())


# B CATEGORICAL VALUES

# Encoding vehicle makes
encoder = TargetEncoder(smoothing=8, min_samples_leaf=5)
CO2_raw_final['Mk'] = encoder.fit_transform(
    CO2_raw_final['Mk'], CO2_raw_final['Enedc (g/km)'])
CO2_raw_final.head(10)
# ***** Dictionary to be created for new instances


CO2_raw_final['Ft'] = CO2_raw_final['Ft'].replace(
    ['PETROL/ELECTRIC'], 'Petrol/Electric')


# Encoding the rest of categorical features
def one_hot_encoding(df, col_name):
    one_hot = pd.get_dummies(df[col_name])
    # Drop the column as it is now encoded:
    df = df.drop(col_name, axis=1)
    # Join the encoded data frame:
    return df.join(one_hot)


CO2_prep_enc = one_hot_encoding(CO2_raw_final, ["Mp", "Fm", "Ft"])

# Zeros and Outliers to be processed

CO2_prep_encf = CO2_prep_enc.replace(
    {'W (mm)': {0: statistics.median(CO2_prep_enc['W (mm)'])}})
CO2_prep_encf['W (mm)'].value_counts()
CO2_prep_encf['ec (cm3)'].describe()

CO2_prep_encf2 = CO2_prep_encf.replace(
    {'ec (cm3)': {7993: statistics.median(CO2_prep_encf['ec (cm3)'])}})
CO2_prep_encf2['ec (cm3)'].describe()


CO2_prep_encf2.to_csv('preprocessed_dataframe.csv', index=False)
