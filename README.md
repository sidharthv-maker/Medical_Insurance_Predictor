# Medical Insurance Cost Predictor

A Machine Learning project that predicts medical insurance charges based on personal and health-related information.

## Project Overview

This project uses regression models to predict insurance charges using features such as age, sex, BMI, number of children, smoking status, and region.

The project also includes basic feature engineering to help the model understand high-cost cases better.

## Dataset

The dataset used is `insurance.csv`.

Main columns:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

Target variable:

```
charges
```
## Example Input
```
Enter age: 45
Enter sex: male
Enter BMI: 32.5
Enter number of children: 2
Are you a smoker? (yes/no): yes
Enter region: South
```
## Example Output
```
Predicted Insurance Charge: $42150.75
```
## Future Improvements
```
Compare Linear Regression, Random Forest, and XGBoost
Add cross-validation results
Save the trained model using joblib
Create a separate prediction file
```
