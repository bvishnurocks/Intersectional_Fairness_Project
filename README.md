# Fairness Analysis Framework

A comprehensive framework for analyzing and mitigating algorithmic bias across multiple datasets using machine learning fairness metrics.

## 📊 Features

- **Multi-Dataset Support**: Works with Adult, COMPAS, and German Credit datasets
- **Multiple Classifiers**: Implements Random Forest, SVM, and Logistic Regression
- **Fairness Metrics**: Calculates SPD, EOD, AOD, and worst-case metrics
- **Interactive Visualizations**: Comprehensive plots and heatmaps
- **Performance Analysis**: Complete performance metrics and comparisons

## 🛠 Installation

### Prerequisites
- Python 3.9 or higher
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fairness-analysis.git
cd fairness-analysis
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Place your datasets in the `datasets/` directory:
   - `adult_processed.csv`
   - `compas_processed.csv`
   - `german_processed_mapped.csv`

## 📂 Project Structure

```
fairness-analysis/
├── datasets/
│   ├── adult_processed.csv
│   ├── compas_processed.csv
│   └── german_processed_mapped.csv
├── src/
│   ├── fairness_analysis.py
│   └── visualization.py
└── notebooks/
    └── fairness_analysis.ipynb
```

## 💻 Usage

### Basic Usage

```python
from fairness_analysis import CompleteFairnessAnalysis

# Initialize analyzer
analyzer = CompleteFairnessAnalysis(datasets_dir='datasets')
analyzer.load_datasets()

# Run analysis
results = analyzer.run_analysis()

# Generate visualizations
visualize_all_results(results)
```

### Visualization Options

1. Performance Metrics:
```python
plot_performance_metrics(results)
```

2. Fairness Metrics:
```python
plot_fairness_metrics(results)
```

3. Protected Attributes Analysis:
```python
plot_protected_attributes(results)
```

4. Worst-Case Metrics:
```python
plot_worst_case_metrics(results)
```

## 📈 Metrics

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Fairness Metrics
- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Average Odds Difference (AOD)
- Worst-case metrics for each

## 📊 Sample Visualizations

1. Performance Comparison
   - Bar plots comparing classifier performance across datasets
   - Metric-specific comparisons

2. Fairness Heatmaps
   - Dataset-specific fairness metric visualizations
   - Protected attribute impact analysis

3. Protected Attributes Analysis
   - Comparative analysis of protected attributes
   - Impact on different fairness metrics

4. Worst-Case Analysis
   - Comparison of worst-case scenarios
   - Dataset and classifier-specific analysis

## 🔍 Supported Datasets

### Adult Dataset
- **Target**: Income prediction
- **Protected Attributes**: Sex, Race
- **Format**: CSV with preprocessed features

### COMPAS Dataset
- **Target**: Recidivism prediction
- **Protected Attributes**: Sex, Race
- **Format**: CSV with preprocessed features

### German Credit Dataset
- **Target**: Credit risk assessment
- **Protected Attributes**: Sex, Age
- **Format**: CSV with preprocessed features

## 📝 Example Output

```python
# Sample Results
ADULT Dataset Summary:
Performance Metrics:
Classifier  Accuracy  Precision  Recall      F1
RF          0.8597    0.8548    0.8597   0.8563
SVM         0.8580    0.8516    0.8580   0.8523
LR          0.8580    0.8517    0.8580   0.8526

Fairness Metrics:
Classifier  SPD      EOD      AOD
RF         -0.2038  -0.6360  -0.3541
SVM        -0.1873  -0.5971  -0.3298
LR         -0.1897  -0.6020  -0.3331
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- scikit-learn team for machine learning implementations
- AIF360 toolkit for fairness metrics inspiration
- Visualization libraries: matplotlib and seaborn

## 📚 References

- [Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf)
- [Machine Learning Fairness](https://developers.google.com/machine-learning/fairness-overview/)
- [AIF360 Documentation](https://aif360.readthedocs.io/)

## 🔗 Additional Resources

- [Project Wiki](https://github.com/yourusername/fairness-analysis/wiki)
- [Issue Tracker](https://github.com/yourusername/fairness-analysis/issues)
- [Documentation](https://yourusername.github.io/fairness-analysis)
