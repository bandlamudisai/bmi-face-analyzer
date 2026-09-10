# Face BMI Estimator

Predicts height, weight, and BMI from a face photo, and classifies the result as
Skinny/Underweight, Normal, Overweight, or Obese.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — upload a photo, click Predict |
| `evaluate.py` | Checks model accuracy against labeled test photos |
| `height_predictor.model` | Trained height regressor |
| `weight_predictor.model` | Trained weight regressor |
| `bmi_predictor.model` | Trained BMI regressor |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages Streamlit Cloud needs to build `dlib` |

The three `.model` files come from running `BMI_prediction_runnable.ipynb` in
Google Colab and downloading the output.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Evaluate the models

```
python evaluate.py --csv "BMI data - Sheet1.csv" --photos height_weight_test --models .
```

This prints predicted vs actual height/weight/BMI for each test photo, plus
overall R² and MSE for each target.




## How it works

1. `face_recognition` detects the face and converts it into a 128-number embedding.
2. Three separate regression models (Kernel Ridge, trained on log-transformed
   targets) map that embedding to height, weight, and BMI.
3. The Streamlit app runs the same embedding step on your uploaded photo, then
   applies the three models and shows the results plus a BMI category.

## Limitations

The training set is a few hundred celebrity photos — predictions are rough
estimates, not medical measurements, and accuracy will vary a lot by photo
angle, lighting, and how well-represented a face type is in the training data.
