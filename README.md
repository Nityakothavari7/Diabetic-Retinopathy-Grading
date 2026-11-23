# Diabetic Retinopathy Grading Using Machine Learning

A machine learning project focused on automated **Diabetic Retinopathy (DR) severity grading** using retinal fundus images, handcrafted image features, lesion-based analysis, and classical machine learning algorithms.

This repository brings together experiments conducted using two retinal fundus image datasets:

- **IDRiD** – Indian Diabetic Retinopathy Image Dataset
- **APTOS 2019** – Blindness Detection Dataset

The project explores the complete workflow, from retinal image preprocessing and feature extraction to model development, evaluation, and Raspberry Pi deployment.

---

## 1. Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that affects the blood vessels in the retina. Early identification and severity assessment are important for preventing vision impairment and blindness.

The objective of this project is to explore machine learning approaches for analysing retinal fundus images and grading the severity of diabetic retinopathy.

Rather than relying entirely on deep learning, this project investigates traditional machine learning techniques supported by handcrafted image features, retinal lesion analysis, and image processing.

### Key Objectives

- Analyse retinal fundus images and their characteristics.
- Perform image preprocessing and enhancement.
- Extract handcrafted image and lesion-related features.
- Investigate feature preprocessing and scaling techniques.
- Develop and compare classical machine learning models.
- Evaluate model performance using appropriate classification metrics.
- Explore deployment of the trained model on Raspberry Pi.

---

## 2. Datasets

The project includes experiments using two publicly available retinal fundus image datasets.

### A. IDRiD – Indian Diabetic Retinopathy Image Dataset

The IDRiD dataset is used for retinal image exploration, feature extraction, lesion-based analysis, and machine learning experiments.

The workflow includes exploratory data analysis, image preprocessing, handcrafted feature extraction, lesion-related feature extraction, and model development.

#### 1. Exploratory Data Analysis

- Initial dataset inspection and organisation.
- Analysis of diabetic retinopathy grade distribution.
- Examination of diabetic macular edema risk labels.
- Visualisation of sample retinal fundus images.
- Exploration of dataset characteristics.

#### 2. Image Preprocessing

- Retinal image preprocessing and preparation.
- Image enhancement.
- Feature organisation.
- Data preprocessing and scaling for machine learning.

#### 3. Feature Extraction

The IDRiD experiments investigate different feature extraction approaches, including:

- Handcrafted retinal image features.
- Lesion-related features.
- BPP feature representation.
- PHOG (Pyramid Histogram of Oriented Gradients) features.

These representations are used to convert retinal image information into structured numerical features for machine learning.

#### 4. Machine Learning

- Training and testing data preparation.
- Experiments using extracted image features.
- Lesion-based machine learning.
- Model development and evaluation.
- Visualisation of experimental outputs.

#### IDRiD Notebooks

| Notebook | Description |
|---|---|
| `Initial_Data_analysis.ipynb` | Initial dataset exploration and analysis |
| `Initial_Data_preprocessing.ipynb` | Initial data preprocessing |
| `Part_1_Feature_Extraction.ipynb` | Handcrafted image feature extraction |
| `Part_1_Lesion_Feature_Extraction.ipynb` | Lesion-related feature extraction |
| `Part_2_Preprocessing_and_Scaling.ipynb` | Feature preprocessing and scaling |
| `PART_2_LESION-BASED-ML.ipynb` | Machine learning using lesion-based features |
| `Model.ipynb` | Model development and experiments |
| `Visualization.ipynb` | Visualisation of dataset and results |

#### IDRiD Results and Outputs

The `IDRiD/results/` directory contains:

- Extracted feature datasets in CSV format.
- Training and testing feature matrices.
- BPP and PHOG feature arrays (`.npy`).
- Training and testing labels.
- Merged label information.
- Diabetic retinopathy grade distribution plots.
- Sample retinal images grouped by DR grade.
- Diabetic macular edema risk visualisations.

---

### B. APTOS 2019 – Blindness Detection Dataset

The APTOS 2019 dataset is used to explore a CPU-oriented diabetic retinopathy grading pipeline based on handcrafted image features and classical machine learning.

The documented workflow uses **3,662 retinal fundus images**.

#### Dataset Distribution

| Dataset split | Number of images |
|---|---:|
| Training | 2,930 |
| Validation | 366 |
| Testing | 366 |
| **Total** | **3,662** |

The classification task involves five diabetic retinopathy severity grades:

| Grade | Description |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

#### 1. Image Preprocessing

The APTOS pipeline includes:

- Retinal image cropping and resizing to 768 × 768.
- Illumination normalisation using robust z-map processing.
- Optic disc localisation.
- Image quality measurements.
- Structural image hashing.
- Duplicate detection and data leakage auditing.

#### 2. Handcrafted Feature Extraction

A total of **30 handcrafted features** are explored across five feature categories.

| Feature category | Number of features |
|---|---:|
| Colour features | 6 |
| GLCM texture features | 5 |
| Vessel-like features | 5 |
| Red-lesion candidate features | 8 |
| Bright-region candidate features | 6 |
| **Total** | **30** |

These features capture different characteristics of retinal fundus images and provide structured representations for subsequent machine learning.

#### 3. Duplicate and Data Leakage Analysis

A duplicate audit was performed to identify duplicated images and potential data leakage across dataset splits.

The documented audit examined:

- 3,662 images.
- 168 verified duplicate pairs.
- 53 duplicate groups spanning dataset splits.
- 39 groups with conflicting grade labels.

This analysis helps identify potential overlap between training, validation, and testing data.

#### 4. Machine Learning Models

The following classical machine learning algorithms were explored:

- Logistic Regression.
- Linear Support Vector Machine (SVM).
- RBF Support Vector Machine.
- Random Forest.
- K-Nearest Neighbours (KNN).

A partial-oversampling experiment was also conducted with Linear SVM to investigate the effect of class imbalance.

#### APTOS Experimental Results

The following results are reported in the project documentation.

| Model | Validation Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 73.22% | 0.5839 |
| Linear SVM | 78.69% | 0.6141 |
| RBF SVM | 75.68% | 0.5925 |
| Random Forest | 81.97% | 0.6062 |
| KNN | 76.50% | 0.5631 |
| Linear SVM with partial oversampling | 78.14% | 0.6269 |

The partial-oversampling Linear SVM experiment also reported a balanced accuracy of **0.6435**.

A provisional Linear SVM baseline reported:

- Test accuracy: 78.96%
- Test macro F1-score: 0.5996

The partial-oversampling model was not evaluated on the test set. The provisional baseline test results should therefore be interpreted separately from the validation comparisons.

---

## 3. Final Model and Raspberry Pi Deployment

The project includes a Raspberry Pi deployment component containing inference-related scripts, testing utilities, execution logs, and the final model deployment work.

The objective is to investigate the practical execution of the trained machine learning model on resource-constrained edge hardware.

### Deployment Components

**Inference Demonstration**

The `raspberry_pi_deployment/inference_demo/` directory contains inference demonstration scripts, including `random_test.py`.

**Testing**

The `raspberry_pi_deployment/tests/` directory contains testing utilities, including `test_dedup.py`.

**Execution Logs**

The `raspberry_pi_deployment/logs/` directory contains:

- `aptos_test_extraction.log`
- `aptos_train_extraction.log`
- `aptos_val_extraction.log`
- `baseline_training.log`

These logs document the corresponding feature extraction and baseline training experiments.

The final trained model and associated deployment files are maintained under the Raspberry Pi deployment component.

---

## 4. Repository Structure

```text
Diabetic-Retinopathy-Grading/
│
├── IDRiD/
│   ├── notebooks/
│   │   ├── Initial_Data_analysis.ipynb
│   │   ├── Initial_Data_preprocessing.ipynb
│   │   ├── Model.ipynb
│   │   ├── PART_2_LESION-BASED-ML.ipynb
│   │   ├── Part_1_Feature_Extraction.ipynb
│   │   ├── Part_1_Lesion_Feature_Extraction.ipynb
│   │   ├── Part_2_Preprocessing_and_Scaling.ipynb
│   │   └── Visualization.ipynb
│   │
│   ├── results/
│   │   ├── X_test_BPP_PHOG.npy
│   │   ├── X_train_BPP_PHOG.npy
│   │   ├── extracted_idrid_features.csv
│   │   ├── idrid_distribution.png
│   │   ├── idrid_labels_merged.csv
│   │   ├── sample_images_dme_risk.png
│   │   ├── sample_images_dr_grades.png
│   │   ├── sample_images_dr_grades_grid.png
│   │   ├── test_features.csv
│   │   ├── test_lesion_features.csv
│   │   ├── train_features.csv
│   │   ├── train_lesion_features.csv
│   │   ├── y_test_BPP_PHOG.npy
│   │   └── y_train_BPP_PHOG.npy
│   │
│   └── DATASET_LICENSE.txt
│
├── APTOS/
│   ├── notebooks/
│   │   ├── 01_drgrade_v2_visual_inspection.ipynb
│   │   ├── audit_duplicates.py
│   │   ├── extract_features.py
│   │   ├── make_splits.py
│   │   ├── train_baselines.py
│   │   └── visualize_pipeline.py
│   │
│   ├── results/
│   │   ├── aptos_test_features_v2.csv
│   │   ├── aptos_train_features_v2.csv
│   │   └── aptos_val_features_v2.csv
│   │
│   └── src/
│       └── drgrade/
│           ├── __init__.py
│           ├── config.py
│           ├── datasets.py
│           ├── debug.py
│           ├── dedup.py
│           ├── features.py
│           └── preprocess.py
│
├── raspberry_pi_deployment/
│   ├── inference_demo/
│   │   └── random_test.py
│   │
│   ├── logs/
│   │   ├── aptos_test_extraction.log
│   │   ├── aptos_train_extraction.log
│   │   ├── aptos_val_extraction.log
│   │   └── baseline_training.log
│   │
│   └── tests/
│       └── test_dedup.py
│
└── README.md
```

*Note: The repository tree lists the files confirmed in the shared GitHub directory listings. Add any additional final model files under `raspberry_pi_deployment/` when documenting their exact names.*

---

## 5. Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming language | Python |
| Development environment | Jupyter Notebook |
| Numerical computing | NumPy |
| Data manipulation | Pandas |
| Image processing | OpenCV |
| Machine learning | Scikit-learn |
| Data visualisation | Matplotlib |
| Image features | GLCM, BPP, PHOG, handcrafted features |
| Deployment platform | Raspberry Pi |

---

## 6. Installation and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nityakothavari7/Diabetic-Retinopathy-Grading.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Diabetic-Retinopathy-Grading
```

### Step 3: Install Dependencies

```bash
pip install numpy pandas opencv-python scikit-learn matplotlib seaborn jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Navigate to the relevant dataset directory and open the notebook you want to explore.

### Dataset Setup

The original retinal fundus images are not redistributed in this repository.

Download the datasets from their respective official sources and organise them according to the file paths expected by the notebooks and scripts.

---

## 7. Limitations and Future Scope

Potential directions for further development include:

- Improving classification performance across underrepresented DR grades.
- Investigating more robust lesion detection and feature extraction methods.
- Evaluating models using additional metrics and independent test data.
- Optimising inference speed and memory usage on Raspberry Pi.
- Improving the deployment pipeline and documenting hardware performance.
- Exploring more generalisable feature representations for retinal image analysis.

Lesion candidate features are heuristic image-processing features and should not be interpreted as clinically validated lesion segmentation.

---

## 8. Dataset Attribution and Licensing

The IDRiD dataset is subject to the licence specified in `IDRiD/DATASET_LICENSE.txt`. Please refer to the file for attribution and usage requirements.

The original IDRiD and APTOS retinal image datasets are not redistributed in this repository. Users must obtain the datasets independently and follow their respective terms of use.

---

## 9. Disclaimer

This project is intended for academic and research purposes.

The implemented models and experiments are not a clinically validated diagnostic system. Their outputs should not be used as a substitute for professional ophthalmological examination, diagnosis, or medical advice.

---

## 10. Author and Repository

**GitHub:** [Nityakothavari7](https://github.com/Nityakothavari7)

**Project Repository:** [Diabetic-Retinopathy-Grading](https://github.com/Nityakothavari7/Diabetic-Retinopathy-Grading)