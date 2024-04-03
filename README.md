# ML-Project

This project contains the code for a machine learning model that classifies images as human or AI-generated. It consists of three distinct models, a Convolutional Neural Network (CNN), a Support Vector Machine (SVM), and a Generative Adversarial Network (GAN). It is NOT an ensemble model, but an investigation into the performance of different models on the same dataset.

## Repository Struture
- `main.py` : Main script to run the project
- `src` : Source Folder
  - `cnn` : CNN model
  - `svm` : SVM model
    - `model.py`: Primary code for the SVM model, containing preprocessing, training, and testing functions
  - `gan` : GAN model
- `datasets` : Folder containing the dataset utilities for training and testing
  - `fake_dataset.py` : Class representing the streamed fake dataset - identical data types as real dataset
  - `real_dataset.py` : Class representing the streamed real dataset - identical data types as fake dataset
  - `Train_GCC-training.tsv` : Training dataset, consists of URLs and captions
  - `Validation_GCC-1.1.0-Validation.tsv` : Validation dataset, consists of URLs and captions
  - `ataset_InfImagine___fake_image_dataset_default_0.0.0_4edf8865bd54ae2acf4708b709b74b6086b61cba.lock` : Fake image dataset, small file containing information retrieved from the HuggingFace datasets library

Additional files in the root directory include
- `requirements.txt` : Required packages for the project
- `README.md` : This file!
- `job.sbatch` : Slurm job script for running the project on the ICE cluster
- `Report-XXXXXXX.out` : Output file from running the project on the ICE cluster
- `gitattributes` : Git LFS file for tracking large files - in our case the training TSV
- `gitignore` : Generic gitignore

## Documentation
Report documentation is found in the `astro-docs` branch, with content served on a GH pages site, available [here](https://github.gatech.edu/pages/rwarner31/ML-Project/)