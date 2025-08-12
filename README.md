# Bias_Testing
AI Testing - Bias
# Advanced Bias Testing for Fraud Detection Models

This repository contains a comprehensive bias testing framework for fraud detection models. The script performs extensive bias analysis including basic metrics, advanced fairness metrics, intersectional analysis, counterfactual fairness testing, and SHAP explainability.

## Features

### Basic Bias Testing
- **Selection Rate Analysis**: Rate of positive predictions per group
- **Demographic Parity**: Difference and ratio of selection rates between groups

### Advanced Bias Testing  
- **Equal Opportunity Difference**: True Positive Rate (TPR) difference between groups
- **Predictive Equality**: False Positive Rate (FPR) difference between groups
- **Equalized Odds**: Maximum of TPR and FPR differences
- **False Negative Rate Difference**: FNR difference between groups
- **Theil Index**: Inequality measure across groups

### Intersectional Analysis
- Analysis of bias at intersections of multiple sensitive attributes
- Comprehensive metrics across Customer_Type × Transaction_Type combinations

### Counterfactual Fairness Testing
- Tests prediction changes when flipping sensitive attribute values
- Individual counterfactual analysis for each sensitive attribute
- Combined counterfactual analysis for multiple attributes

### SHAP Explainability Audit
- Feature importance analysis using SHAP values
- Quantifies reliance on sensitive attributes
- Group-wise SHAP value distribution analysis

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- xgboost>=1.4.0
- fairlearn>=0.7.0
- shap>=0.40.0
- scipy>=1.7.0

## Usage

### Command Line Interface

```bash
python advanced_bias_testing.py --train path/to/train.csv --test path/to/test.csv [options]
```

### Arguments

- `--train`: Path to training CSV file (required)
- `--test`: Path to testing CSV file (required)
- `--model`: Model type ('randomforest' or 'xgboost', default: 'randomforest')
- `--output`: Output file for bias report (default: 'bias_report.csv')

### Example

```bash
python advanced_bias_testing.py --train sample_train.csv --test sample_test.csv --model randomforest --output my_bias_report.csv
```

## Data Format

Your CSV files must contain the following columns:

- `Transaction_Date`: Date of transaction
- `Transaction_Time`: Time of transaction  
- `Customer_ID`: Unique customer identifier
- `Customer_Type`: Sensitive attribute #1 (e.g., 'Premium', 'Standard', 'Basic')
- `Transaction_Type`: Sensitive attribute #2 (e.g., 'Online', 'ATM', 'POS')
- `Transaction_Amount`: Transaction amount
- `Is_Fraudulent`: Binary label (0 = legitimate, 1 = fraudulent)

## Sample Data Generation

To test the script with sample data:

```bash
python create_test_data.py
```

This generates `sample_train.csv` and `sample_test.csv` with intentional bias patterns for testing.

## Output

The script produces:

1. **Console Output**: Comprehensive bias analysis summary with:
   - Model performance metrics
   - Basic bias metrics with interpretations
   - Advanced fairness metrics
   - Intersectional bias analysis
   - Counterfactual fairness results
   - SHAP explainability insights

2. **CSV Report**: Detailed bias report saved to specified output file containing:
   - All calculated metrics
   - Interpretations and thresholds
   - Group-wise breakdowns

## Interpretation Guide

### Bias Severity Levels

**Demographic Parity Difference:**
- ✅ Low bias: < 0.1
- ⚠️ Moderate bias: 0.1-0.2  
- 🚨 High bias: > 0.2

**SHAP Sensitive Attribute Importance:**
- ✅ Low reliance: < 10%
- ⚠️ Moderate reliance: 10-25%
- 🚨 High reliance: > 25%

**Counterfactual Fairness:**
- Lower percentage of changed predictions indicates better individual fairness
- Mean prediction change indicates magnitude of bias

### Key Metrics Explained

- **Equal Opportunity**: Ensures equal TPR across groups (important for fraud detection)
- **Predictive Equality**: Ensures equal FPR across groups  
- **Equalized Odds**: Combines both TPR and FPR fairness
- **Theil Index**: Measures inequality distribution (0 = perfect equality)

## Framework Architecture

The `BiasTestingFramework` class is modular with separate methods for:

- `load_data()`: CSV loading with validation
- `preprocess_data()`: Feature engineering and encoding
- `train_model()`: Model training with RandomForest or XGBoost
- `calculate_basic_bias_metrics()`: Selection rates and demographic parity
- `calculate_advanced_bias_metrics()`: Advanced fairness metrics
- `analyze_intersectional_bias()`: Multi-attribute bias analysis
- `perform_counterfactual_testing()`: Individual fairness testing
- `perform_shap_analysis()`: Explainability analysis
- `generate_bias_report()`: Comprehensive reporting

## Best Practices

1. **Data Quality**: Ensure your data has sufficient samples per group for reliable analysis
2. **Model Selection**: Try both RandomForest and XGBoost to compare bias patterns
3. **Threshold Setting**: Consider your use case when interpreting bias severity levels
4. **Intersectional Analysis**: Pay special attention to small intersectional groups
5. **Counterfactual Results**: High prediction changes indicate potential unfairness

## Limitations

- SHAP analysis is limited to tree-based models (RandomForest, XGBoost)
- Intersectional analysis requires sufficient samples per intersection (>10)
- Counterfactual testing assumes binary flipping of categorical attributes
- Some advanced metrics require at least 2 groups per sensitive attribute

## Contributing

Feel free to extend the framework with additional bias metrics or model types. The modular design makes it easy to add new analysis methods.

## License

This project is provided as-is for educational and research purposes.
