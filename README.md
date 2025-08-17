# Estimation of daily surface air temperatures from EOS-MODIS data: Methodology and application for climate trends and extreme event analysis

## Objectives
Surface Air Temperature (SAT) is an Essential Climate Variable (ECV) widely used for weather monitoring and climate analysis. However, it cannot be directly retrieved from satellite observations. In a previous project, we found that the XGBoost algorithm was the most accurate method for estimating instantaneous SAT. Thus, in this project applies XGBoost not only to estimate instantaneous SAT but also to retrieve maximum, minimum, and mean SAT values. XGBoost models where used to obtain maps to study SATs trends over the Spanish Iberian Peninsula and to detect extreme events associated with temperature such as tropical and ecuatorial nights.

## Project structure

This project was part of my second article. It was initially developed rapidly, Due to strict project deadlines, for use in a local environment. However, I am currently restructuring and improving the code to ensure better organization, OS compatibility, and usability. Updated modules will be uploaded progressively, so that anyone can download, run, and adapt the scripts for their own research or applications. 

Data files are not upload because AEMET's data is not public. For more information do not hesitate to contact me.

The project structure will be continuously updated as new files are added.

## Project Status
This is a finished project, but the current version is being cleaned and prepared for public release. Paths, modularity, and documentation are being improved for OS-independence and reproducibility and they will be uploaded progressively.

**Current project structure**

```
.
├── data/
│   ├── aemet 
│   ├── geodata
│   └── modis
├── outputs/
│   ├── modeling/
│   │   ├── feature_selection
│   │   ├── models
│   │   └── parameters
│   └── preprocessing/
│       ├── aemet
│       ├── databases
│       └── modis
├── scripts/
│   ├── modeling/
│   │   └── models_training.py # script to train models
│   └── preprocessing/
│       ├── create_databases.py # script to create final databases from preprocessed inputs
│       └── preprocess_input_data.py # script to process the different inputs used in this study
├── LICENSE
└── README.md
```

## Publication
*Joan Vedrí, Raquel Niclòs, Lluís Pérez-Planells, Enric Valor, María Yolanda Luna, Fernando Belda* (Under Review). **Estimation of daily surface air temperatures from EOS-MODIS data: Methodology and application for climate trends and extreme event analysis**. *IEEE Transactions on Geoscience and Remote Sensing.* 
