# Housing Prices Prediction Project

## Overview
This project analyzes California housing data to predict median house values using various features such as location, housing characteristics, and demographics.

## Dataset Description
- **longitude**: Distance west (higher = farther west)
- **latitude**: Distance north (higher = farther north)
- **housing_median_age**: Median age of houses in block (lower = newer)
- **total_rooms**: Total rooms in the block
- **total_bedrooms**: Total bedrooms in the block
- **population**: Number of residents in the block
- **households**: Number of households in the block
- **median_income**: Median household income (tens of thousands USD)
- **median_house_value**: Median house value (USD)
- **ocean_proximity**: Categorical location relative to ocean/bay

## Steps Taken

1. Loaded and cleaned data by dropping rows with missing values.
2. Explored feature distributions and correlations.
3. Applied log transformations to skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`) to normalize data.
4. One-hot encoded `ocean_proximity` categorical variable.
5. Created new features:
   - `bedroom_ratio` = total_bedrooms / total_rooms
   - `household_rooms` = total_rooms / households
6. Split data into training (80%) and test (20%) sets.
7. Scaled numeric features with `StandardScaler`.
8. Trained and evaluated models:
   - Linear Regression (baseline)
   - Random Forest Regressor (better performance)
9. Performed hyperparameter tuning on Random Forest using `GridSearchCV`.
10. Achieved final test \( R^2 \) score of approximately 0.83.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
- Install dependencies with: pip install pandas numpy matplotlib seaborn scikit-learn

- Place `housing.csv` in the working directory.
- Run the Python script or Jupyter notebook to preprocess data, train models, and evaluate.

## Results
- Log transformations helped reduce skewness.
- One-hot encoding allowed inclusion of categorical data.
- Random Forest with tuning significantly outperformed Linear Regression.
- Final model explains ~83% of variance in house prices on test data.

## Future Improvements
- Try other regression models (e.g., Gradient Boosting).
- Implement more robust cross-validation.
- Perform feature selection or dimensionality reduction.
- Add domain-specific engineered features.

---

## How to Run
- Install dependencies with:
