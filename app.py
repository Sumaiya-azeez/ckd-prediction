from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import lime
import lime.lime_tabular

app = Flask("CKDAPP")

model = joblib.load("model_outputs/best_model.pkl")

features = [
    "age",
    "red_blood_cells",
    "pus_cell",
    "blood_glucose_random",
    "blood_urea",
    "pedal_edema",
    "anemia",
    "diabetesmellitus",
    "coronary_artery_disease"
]

df = pd.read_csv("dataset/clean9.csv")

train = np.array(df)

limeexplainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=train,
    feature_names=features,
    class_names=["No CKD","CKD"],
    mode="classification"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    vals = []
    for f in features:
        vals.append(float(request.form[f]))

    x = pd.DataFrame([vals], columns=features)

    p = model.predict_proba(x)[0]
    prob = p[1]
    pred = "CKD Detected" if prob >= 0.5 else "No CKD Detected"

    exp = limeexplainer.explain_instance(
        x.iloc[0],
        model.predict_proba,
        num_features=9
    )
    raw_list = exp.as_list()

    table = []
    for feat, weight in exp.as_list():
        table.append({
            "feature": feat,
            "weight": round(weight, 3),
            "abs": abs(weight)
        })
    table = sorted(table, key=lambda r: r["abs"], reverse=True)

    top3 = table[:3]
    summary_parts = []
    for r in top3:
        if r["weight"] > 0:
            summary_parts.append(f"{r['feature']} increases CKD risk ({r['weight']:+})")
        else:
             summary_parts.append(f"{r['feature']} decreases CKD risk ({r['weight']:+})")
    summary = "; ".join(summary_parts)

    return render_template(
        "resultlime.html",
        prediction=pred,
        probability=round(prob,3),
        explanation=table,
        summary=summary
    )

app.run(debug=True)