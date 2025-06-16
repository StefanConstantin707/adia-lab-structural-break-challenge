# ADIA Lab Structural Break Challenge Solution

This repository contains a solution for the ADIA Lab Structural Break Challenge on CrunchDAO, where the goal is to detect structural breaks in time series data.

## 🎯 Competition Overview

- **Task**: Binary classification - determine if a structural break occurred at a specified boundary point
- **Data**: Synthetic univariate time series (1,000-5,000 values each)
- **Metric**: ROC AUC score
- **Prize Pool**: $100,000 total

## 📁 Project Structure

```
structural-break-challenge/
├── data/                   # Competition data (auto-managed by crunch-cli)
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   └── utils/             # Utilities and helpers
├── tests/                  # Unit tests
├── experiments/            # Experiment tracking
├── resources/             # Model storage (required by competition)
├── submissions/           # Competition submission files
└── scripts/               # Training and evaluation scripts
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd structural-break-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup CrunchDAO CLI

```bash
# Get your token from the competition page
crunch setup structural-break model-v1 --token <YOUR_TOKEN>
```

### 3. Explore the Data

Start with the exploration notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Train a Model

```bash
# Using the provided script
python scripts/train_model.py

# Or use crunch CLI for local testing
crunch test
```

### 5. Submit Your Solution

```bash
# Test locally first
crunch test

# Then push to CrunchDAO
crunch push --message "Initial XGBoost baseline"
```

## 📊 Solution Approach

### Feature Engineering

The solution uses multiple types of features:

1. **Statistical Features**
   - Mean, std, skewness, kurtosis for each period
   - Differences and ratios between periods

2. **Statistical Tests**
   - T-test for mean differences
   - Levene test for variance differences
   - Kolmogorov-Smirnov test for distribution changes
   - Mann-Whitney U test (non-parametric)

3. **Time Series Features** (to implement)
   - Autocorrelation changes
   - Spectral density
   - Complexity measures

### Models

- **Baseline**: XGBoost with statistical features
- **Advanced**: Ensemble of multiple models
- **Future**: Deep learning approaches (LSTM, CNN)

## 📈 Development Workflow

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Understand data structure
   - Visualize examples
   - Analyze break characteristics

2. **Feature Engineering** (`notebooks/02_feature_engineering.ipynb`)
   - Implement feature extractors
   - Test feature importance
   - Select best features

3. **Model Development** (`notebooks/03_model_experiments.ipynb`)
   - Try different algorithms
   - Tune hyperparameters
   - Validate performance

4. **Final Submission** (`submissions/main.py`)
   - Implement train() and infer() functions
   - Ensure deterministic output
   - Meet time constraints

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Test submission locally:
```bash
crunch test
```

## 📝 Key Files

- `src/data/dataloader.py`: Data loading and preprocessing class
- `submissions/main.py`: Main submission file with train() and infer()
- `notebooks/01_data_exploration.ipynb`: Initial data analysis
- `scripts/evaluate_model.py`: Model evaluation script

## 💡 Tips for Success

1. **Start Simple**: Begin with basic statistical features and XGBoost
2. **Validate Carefully**: Use proper cross-validation to avoid overfitting
3. **Feature Engineering**: This is key - try many different features
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Monitor Time**: Ensure your solution runs within the 15-hour weekly limit

## 🔧 Advanced Techniques to Try

1. **Change Point Detection Algorithms**
   - CUSUM (Cumulative Sum)
   - PELT (Pruned Exact Linear Time)
   - Binary Segmentation

2. **Deep Learning**
   - LSTM autoencoders
   - 1D CNN for pattern recognition
   - Attention mechanisms

3. **Feature Selection**
   - Recursive Feature Elimination
   - Permutation importance
   - SHAP values

## 📚 References

- [CrunchDAO Documentation](https://docs.crunchdao.com/)
- [Structural Break Detection Methods](https://en.wikipedia.org/wiki/Structural_break)
- [Time Series Analysis](https://otexts.com/fpp3/)

## 🤝 Contributing

Feel free to experiment and improve the solution! Key areas for contribution:
- New feature engineering methods
- Alternative model architectures
- Hyperparameter optimization
- Ensemble strategies

## 📄 License

This project is for the ADIA Lab Structural Break Challenge competition.