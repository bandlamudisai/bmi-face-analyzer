import argparse
import glob
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import face_recognition
from sklearn.metrics import mean_squared_error, r2_score


def get_index_of_digit(stem):
    match = re.search(r"\d", stem)
    return match.start(0) if match else len(stem)


def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None


def main(csv_path, photos_dir, models_dir):
    profile_df = pd.read_csv(csv_path)

    all_jpgs = sorted(
        f for f in glob.glob(f"{photos_dir}/*")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    id_path = [(Path(img).stem[:get_index_of_digit(Path(img).stem)], img) for img in all_jpgs]
    image_df = pd.DataFrame(id_path, columns=["UID", "path"])
    data_df = image_df.merge(profile_df, on="UID")

    height_model = joblib.load(f"{models_dir}/height_predictor.model")
    weight_model = joblib.load(f"{models_dir}/weight_predictor.model")
    bmi_model = joblib.load(f"{models_dir}/bmi_predictor.model")

    rows = []
    for _, row in data_df.iterrows():
        enc = get_face_encoding(row["path"])
        if enc is None:
            continue
        X = np.expand_dims(enc, axis=0)
        pred_height = float(np.exp(height_model.predict(X)[0]))
        pred_weight = float(np.exp(weight_model.predict(X)[0]))
        pred_bmi = float(np.exp(bmi_model.predict(X)[0]))
        rows.append({
            "UID": row["UID"],
            "path": row["path"],
            "actual_height": row["height"],
            "pred_height": round(pred_height, 2),
            "actual_weight": row["weight"],
            "pred_weight": round(pred_weight, 1),
            "actual_bmi": row["BMI"],
            "pred_bmi": round(pred_bmi, 1),
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    print("\nHeight  R2:", round(r2_score(result_df.actual_height, result_df.pred_height), 4),
          " MSE:", round(mean_squared_error(result_df.actual_height, result_df.pred_height), 4))
    print("Weight  R2:", round(r2_score(result_df.actual_weight, result_df.pred_weight), 4),
          " MSE:", round(mean_squared_error(result_df.actual_weight, result_df.pred_weight), 4))
    print("BMI     R2:", round(r2_score(result_df.actual_bmi, result_df.pred_bmi), 4),
          " MSE:", round(mean_squared_error(result_df.actual_bmi, result_df.pred_bmi), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--models", default=".")
    args = parser.parse_args()
    main(args.csv, args.photos, args.models)
