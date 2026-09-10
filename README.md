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

## Upload to GitHub

1. Create a new empty repository on github.com (no README/license, since you already have them).
2. From the folder containing all the files above:

```
git init
git add .
git commit -m "Face BMI estimator"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

You'll be prompted for GitHub credentials — use a personal access token as the
password (GitHub → Settings → Developer settings → Personal access tokens).

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app", select your repo and branch, set main file to `app.py`.
3. Deploy. First build takes 10–20 minutes because `dlib` compiles from source
   using the packages listed in `packages.txt`.

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
