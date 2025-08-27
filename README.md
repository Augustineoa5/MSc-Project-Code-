# Lightweight Machine Learning-Based Anomalous Signal Detection for Smart Meter Data in Smart Grids

## Overview
This repository contains the implementation of a thesis project focused on developing lightweight machine learning models for real-time anomaly detection in smart meter data within smart grids. The project emphasises efficiency, accuracy, and scalability for resource-constrained edge devices, such as those in IoT-enabled smart grids. It addresses challenges like equipment failures and cyberattacks through anomaly detection.

### Key Components
- **Machine Learning Models**: Implementation and evaluation of lightweight algorithms, including:
  - Decision Trees (base, with optimised version in a separate root folder file)
  - Isolation Forests (base, with adjusted version in a separate root folder file)
  - Random Forests (base, with optimised version in a separate root folder file)
  - Gradient Boosting
- **NB**: All 7 models are trained in the code saved as models.py
- **Dataset Handling**: Preprocessing and analysis of simulated and real-world datasets, including a subset from the [UCI Machine Learning Repository: Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption).
- **Web Application**: A Flask-based web app for sequential model training, external dataset validation, performance evaluation, and visualisation (e.g., confusion matrices, ROC curves). Note: Optimised model versions are not included in the web app files but are available in separate files in the root folder.
- **Edge Deployment Focus**: Models optimised for low latency and minimal resource usage, to be tested in simulated environments like Raspberry Pi 4.

This project serves as a proof-of-concept for real-time anomaly detection in smart grids.

## Features
- **Anomaly Detection**: Statistical outlier detection using rolling means and standard deviations, supporting class-imbalanced datasets.
- **Model Optimisation**: Hyperparameter tuning with GridSearchCV, feature subset analysis, and cross-validation.
- **Web Interface**: A user-friendly 5-step workflow for data upload, model training, testing, and comparative analysis.
- **Visualisations**: Automatically generated plots for model performance metrics (e.g., confusion matrices, ROC curves).
- **Efficiency Metrics**:
  - Inference time: <1 second
  - Memory usage: <5 MB
  - Accuracy: Up to 95% for top-performing models

## Repository Structure
```
smart-meter-anomaly-detection/
├── app.py                          # Flask application for the web app
├── utils.py                        # Utility functions for data preprocessing, model training, and evaluation
├── templates/                      # HTML templates for the web interface
│   └── index.html                  # Main page with 5-step workflow
├── static/                         # Static assets
│   └── css/
│       └── styles.css              # CSS styles for the interface
├── datasets/                       # Directory for datasets
│   └── household_consumption.csv   # Preprocessed dataset (subset of UCI data)
├── visualizations/                 # Directory for generated plots (e.g., confusion matrices, ROC curves)
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation file
└── venv/                           # Virtual environment (ignored in .gitignore)
```

**Note**: The `venv/` folder is ignored in `.gitignore` to avoid committing virtual environments. Generated visualizations are stored dynamically in `/visualizations/`.

## Requirements
- **Python**: 3.8+ (tested on 3.12.3)
- **Dependencies** (listed in `requirements.txt`):
  - Flask
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

No additional packages are required beyond these standard scientific computing libraries.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aowusu-asampong/smart-meter-anomaly-detection.git
   cd smart-meter-anomaly-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```
   - On Windows: `venv\Scripts\activate`
   - On Unix/Mac: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Download the full UCI dataset and place a subset in `/datasets/` as `household_consumption.csv`. The code handles preprocessing for 7,500 records.

## Usage
### Running the Web Application
1. Start the Flask server:
   ```bash
   python app.py
   ```
   Alternatively:
   ```bash
   export FLASK_APP=app.py
   flask run
   ```
2. Access the app at `http://127.0.0.1:5000` in your browser.
3. Follow the 5-step workflow:
   - **Step 1**: Upload a training dataset (CSV format, e.g., `household_consumption.csv`).
   - **Step 2**: Select and train models sequentially (e.g., Decision Tree, Random Forest).
   - **Step 3**: Upload a test dataset for external validation.
   - **Step 4**: Run tests on trained models.
   - **Step 5**: View comparative results, metrics, and visualizations.

The app supports persistent storage of models and results across sessions. For detailed methodology, refer to the thesis document (not included in the repository; see abstract in the overview).

### Running Models Standalone
Use `utils.py` for standalone scripting:
```python
import utils
utils.train_model('DecisionTree', dataset_path='datasets/household_consumption.csv')
```
Available functions include `preprocess_data()`, `train_model()`, and `evaluate_model()`. Optimized model versions are available in separate files in the root folder.

## Example Dataset
The included `household_consumption.csv` is a preprocessed subset (7,500 records) with features:
- `datetime`
- `global_active_power` (consumption)
- `global_reactive_power`
- `voltage`
- `global_intensity`
- `sub_metering_1/2/3`
- `hour`
- `day_of_week`
- `synthetic_temperature`

Anomalies are labeled based on deviations (>2 standard deviations from the rolling mean).

## Results Highlights
- **Best Model**: Optimized Decision Tree with Time Features (94.13% accuracy, 0.6526 F1-score for anomalies).
- **Ensemble Alternatives**: Gradient Boosting (95% accuracy) offers higher recall but is more resource-intensive.
- **Visualizations**: Generated on-the-fly and stored in `/visualizations/`.

## Limitations and Future Work
- Tested in simulated environments; real-world edge device validation is recommended.
- Future enhancements could include hybrid models or live data streaming.
- Ethical considerations: Ensure data privacy compliance when using real smart meter data.

## Citation
If using this code in your research, please cite the thesis:
```
Augustine Owusu Asampong. (2025). Lightweight Machine Learning-Based Anomalous Signal Detection for Smart Meter Data in Smart Grids. [Thesis]. Robert Gordon University.
```

## License
This project is not yet licensed. To add a license (e.g., MIT License), create a `LICENSE` file in the repository root with the appropriate license text. For guidance, refer to [choosealicense.com](https://choosealicense.com/).

## Contact
For questions or contributions, open an issue on GitHub or contact Augustine Owusu Asampong at [a.owusu-asampong@rgu.ac.uk](mailto:a.owusu-asampong@rgu.ac.uk).
