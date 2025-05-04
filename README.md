# Conv1D + LSTM Call Volume Forecasting

## Overview

This project demonstrates how a hybrid Conv1D + LSTM deep learning model was used to forecast daily inbound call volumes for a U.S.-based luxury travel company. Accurate forecasts help optimize staffing decisions and support operations across Sales and Customer Service departments.

### Highlights

* **Architecture**: Conv1D for pattern extraction, LSTM for sequential memory
* **Libraries**: TensorFlow/Keras, scikit-learn, pandas, holidays
* **Performance**: Achieved \~30% improvement in forecast accuracy over Excel-based trendline forecasting baselines
* **Validation MAE**: \~0.0485 on scaled test data

## Business Context

Call volumes fluctuate due to seasonal trends, day-of-week effects, and holiday impacts. Traditional manual or linear models lacked responsiveness. The Conv1D + LSTM model captured nonlinear trends and temporal dependencies, significantly improving resource planning accuracy.

## Data & Features

Original call volume data is proprietary and not included. The model was trained on confidential datasets with the following engineered features:

* `date`: Date of the record (daily granularity)
* `Total_Presented`: Number of inbound calls on that day (target variable)
* **Customer Potential Proxy**:

  * **Sales model**: A forward-looking proxy indicating sales potential based on cruise inventory
  * **Service model**: A short-term proxy for post-departure service demand based on recent travel activity
* `PaymentsDue`: (Service only) Count of customer payments due on a given day (used as a proxy for follow-up service demand)
* `Holiday_A`: Binary indicator for full-company closure days (no calls expected)
* `Holiday_B`: Binary indicator for reduced-staff holidays (limited operations)
* `weekday`: Day of the week (0 = Monday, 6 = Sunday)
* `week_number`: ISO week number (1–52)
* `month`: Calendar month (1–12)
* `monthly_sum`: Total call volume for the current month (smoothing/trend feature)
  
Note: The implementation in the provided notebooks utilizes standard US public holidays (e.g., Thanksgiving, Christmas Day for Holiday_A) as concrete examples for these feature flags.
The data was normalized using `MinMaxScaler`, and sequences were generated with a 30-day sliding window.

## Model Architecture

* **Conv1D Layer**: Extracts short-term call volume trends
* **MaxPooling1D**: Downsamples features for generalization
* **LSTM Layer**: Captures long-term dependencies
* **Dense Layer**: Produces forecast output
* **Regularization**: L2 penalties and dropout layers reduce overfitting
* **EarlyStopping**: Stops training when validation loss stops improving

## Department-Specific Variants

* `direct_department_forecast.ipynb` (Sales): Predicts sales-related call volumes using a forward-looking sales potential proxy.
* `service_department_forecast.ipynb` (Service): Predicts service-related calls using a short-term demand proxy and payment-based indicators.

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook notebooks/direct_department_forecast.ipynb
# or
jupyter notebook notebooks/service_department_forecast.ipynb
```

## Repository Structure

```
├── notebooks/
│   ├── direct_department_forecast.ipynb
│   └── service_department_forecast.ipynb
├── data/
│   └── sample_data.csv (optional demo input)
├── src/ (optional Python scripts)
├── saved_models/ (optional saved model artifacts)
├── requirements.txt
└── README.md
```

## Results

The Conv1D + LSTM model improved forecasting accuracy by \~30% compared to Excel-based trendline forecasting baselines. The best model achieved an MAE of \~0.0485 on scaled data, which translated into substantial improvements in daily staffing and scheduling decisions.

## Limitations

* Sales call volumes are inherently noisier and less predictable than Service calls
* Model performance depends on the consistency of historical patterns

## Future Enhancements

* Add external features (e.g., campaign flags, weather, macro trends)
* Explore Transformer or attention-based time series models
* Containerize and deploy as a microservice with REST API endpoints

## Author

**Xichun (Harrison) Han**
AI & Data Strategy Manager @ Viking Cruises
[LinkedIn](https://www.linkedin.com/in/xichun-han)

---

**Disclaimer**: All data included is anonymized or simulated for illustrative purposes only.
