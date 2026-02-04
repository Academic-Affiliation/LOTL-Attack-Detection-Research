# LOTL-Attack-Detection-APT

## Overview
This repository contains machine learning models and datasets for detecting Living-off-the-Land (LOTL) attacks and APT (Advanced Persistent Threat) activities. The project focuses on identifying malicious behavior using alternative attack syntax and obfuscation techniques.

## Dataset
The project includes multiple datasets:
- **advanced_lotl_alternative_syntax_dataset.csv**: LOTL attacks with alternative syntax variations
- **advanced_lotl_generator.py**: Generator for creating advanced LOTL attack samples
- **balanced_combined_lotl_dataset.csv**: Balanced dataset combining LOTL and benign samples
- **balanced_volttyphoon_dataset.csv**: Balanced Volt Typhoon APT dataset
- **lotl_benign_dataset.csv**: Benign system activities for baseline comparison
- **LOTLWorld Benign Attack.csv**: Additional benign activity samples
- **LOLBAS APIs.csv**: LOLBAS (Living off the Land Binaries) API reference

## Code Structure
- **LOTL Attack Detection.ipynb**: Main detection model notebook
- **BERT LOLWTC Detection.ipynb**: BERT-based LOTL detection implementation
- **LOTL Detection Analysis.ipynb**: Comprehensive analysis and evaluation of detection models
- **analyze_dataset_features.py**: Dataset feature analysis and preprocessing
- **advanced_lotl_generator.py**: Advanced LOTL sample generation script

## Models
- BERT-based detection model for LOTL attacks
- Machine learning classifiers for APT activity identification
- Behavioral analysis for suspicious command execution
# LOTL-Attack-Detection-
