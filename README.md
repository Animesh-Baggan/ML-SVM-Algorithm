## Machine Learning Notebooks Collection

This repository contains a curated set of Jupyter notebooks demonstrating core machine learning algorithms and techniques from scratch and with scikit-learn. It is intended for learning, experimentation, and quick reference.

### Contents

- **Projects/**
  - `Bagging.ipynb`: Bagging ensemble basics with examples.
  - `Random Forest - Tuning.ipynb`: Random Forest training and hyperparameter tuning.
  - `Voting_classifier.ipynb`: Hard/soft voting ensembles.
  - `SVM Algorithm.ipynb`: Support Vector Machines with linear and non-linear kernels.
  - `Logistic Regression.ipynb`: Binary classification with logistic regression.
  - `ML Linear Agg.ipynb`: Linear regression aggregation/analysis examples.
  - `ML Multiple Regresion.ipynb`: Multiple linear regression.
  - `ML Regg From Scratch.ipynb`: Linear regression implemented from first principles.
  - `Gradient_Descent_From_scratch.ipynb`: Gradient descent algorithm step-by-step.
  - `test.ipynb`: Scratchpad and quick experiments.
  - Per-topic READMEs: `README_Random_forest.md`, `README_Bagging.txt`, etc.

### Prerequisites

- Python 3.9+ recommended
- pip (or uv/poetry/conda)

### Quickstart

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter and open any notebook in `Projects/`:

```bash
jupyter notebook
```

### Notes and Tips

- Some examples use scikit-learn. With scikit-learn ≥ 1.0, certain imports changed (e.g., use `from sklearn.datasets import make_blobs`).
- If plots don’t display, ensure the first cell contains `%matplotlib inline` and you’ve executed it.
- Randomness: results may vary across runs; many notebooks set a `random_state` for reproducibility.

### Suggested Learning Path

1. `Gradient_Descent_From_scratch.ipynb`
2. `ML Regg From Scratch.ipynb` → `ML Multiple Regresion.ipynb`
3. `Logistic Regression.ipynb`
4. `SVM Algorithm.ipynb`
5. Ensembles: `Bagging.ipynb`, `Random Forest - Tuning.ipynb`, `Voting_classifier.ipynb`

### Project Structure

```
/Projects
  ├── *.ipynb               # Topic-focused notebooks
  ├── README_*.md|txt       # Topic-specific notes
requirements.txt            # Python dependencies
README.md                   # This file
```

### Troubleshooting

- "ModuleNotFoundError": Verify your virtual environment is active and dependencies are installed.
- "ImportError" related to scikit-learn datasets: update imports to modern paths (see Notes and Tips).
- Jupyter can’t find the kernel: reinstall ipykernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=ml-notebooks
```

### License

This repository is provided for educational purposes. If you plan to use it beyond personal learning, please add a license suitable for your needs (e.g., MIT).

### Acknowledgements

- scikit-learn, NumPy, pandas, matplotlib, seaborn, and the broader open-source community.

# Voting Classifier Implementation

A comprehensive implementation of ensemble learning using voting classifiers in machine learning. This project demonstrates how to combine multiple machine learning models to improve prediction accuracy through both hard and soft voting mechanisms.

## 🎯 Overview

This project showcases the power of ensemble learning by implementing voting classifiers that combine multiple base models to achieve better performance than individual classifiers. The implementation covers:

- **Hard Voting**: Simple majority voting among classifiers
- **Soft Voting**: Probability-weighted voting for improved accuracy
- **Weighted Voting**: Custom weight assignments to different classifiers
- **Cross-validation**: Robust performance evaluation using k-fold cross-validation

## 🚀 Features

- **Multiple Base Classifiers**: Logistic Regression, Random Forest, K-Nearest Neighbors
- **SVM Ensemble**: Polynomial kernel SVMs with different degrees
- **Performance Comparison**: Individual vs. ensemble model performance analysis
- **Weight Optimization**: Systematic exploration of different weight combinations
- **Visualization**: Data exploration using seaborn pairplots

## 📊 Datasets Used

1. **Iris Dataset**: Classic classification dataset with 3 species of iris flowers
   - Features: Sepal length, Sepal width, Petal length, Petal width
   - Target: Species classification (0, 1, 2)

2. **Synthetic Classification Dataset**: Generated dataset for SVM ensemble demonstration
   - 1000 samples with 20 features
   - 15 informative features, 5 redundant features

## 🔧 Implementation Details

### Base Classifiers
- **Logistic Regression**: Linear classification model
- **Random Forest**: Ensemble of decision trees
- **K-Nearest Neighbors**: Instance-based learning

### Voting Strategies

#### Hard Voting
```python
vc = VotingClassifier(estimators=estimators, voting='hard')
```
- Simple majority vote among all classifiers
- Each classifier gets equal weight

#### Soft Voting
```python
vc = VotingClassifier(estimators=estimators, voting='soft')
```
- Uses probability predictions from each classifier
- More nuanced than hard voting

#### Weighted Voting
```python
vc = VotingClassifier(estimators=estimators, voting='soft', weights=[i,j,k])
```
- Custom weights for each classifier
- Systematic exploration of weight combinations (1-3 for each classifier)

### SVM Ensemble
- **Polynomial Kernels**: Degrees 1-5
- **Probability Estimation**: Enabled for soft voting
- **Cross-validation**: 10-fold CV for robust evaluation

## 📈 Results

### Individual Classifier Performance (Iris Dataset)
- Logistic Regression: 0.81
- Random Forest: 0.70
- K-Nearest Neighbors: 0.76

### Ensemble Performance
- **Hard Voting**: 0.77
- **Best Weighted Voting**: 0.80 (weights: [3,1,1])

### SVM Ensemble Performance
- **Individual SVMs**: 0.81-0.89 accuracy
- **Ensemble (Soft Voting)**: 0.93 accuracy

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

### Key Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **seaborn**: Statistical data visualization
- **matplotlib**: Plotting library

## 📁 Project Structure

```
Voting_classifier.ipynb          # Main Jupyter notebook
README.md                        # This file
```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Voting_classifier.ipynb
   ```

4. **Download the Iris dataset**
   - Place `iris.csv` in your working directory
   - Or modify the file path in the notebook

## 💡 Key Insights

1. **Ensemble Learning Benefits**: The voting classifier often outperforms individual models
2. **Weight Optimization**: Custom weights can significantly improve ensemble performance
3. **Soft vs Hard Voting**: Soft voting generally provides better results than hard voting
4. **Cross-validation**: Essential for reliable performance estimation

## 🔍 Code Sections

1. **Data Loading & Preprocessing**: Loading iris dataset and label encoding
2. **Exploratory Data Analysis**: Pairplot visualization of features
3. **Base Classifier Training**: Individual model performance evaluation
4. **Voting Classifier Implementation**: Hard and soft voting
5. **Weight Optimization**: Systematic weight combination exploration
6. **SVM Ensemble**: Polynomial kernel SVM combination
7. **Performance Comparison**: Cross-validation results analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Animesh Baggan**

## 📚 References

- [Scikit-learn Voting Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [Ensemble Learning Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

## ⭐ Star the Repository

If you find this project helpful, please consider giving it a star on GitHub!

