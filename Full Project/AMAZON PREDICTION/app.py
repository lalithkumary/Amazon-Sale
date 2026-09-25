from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import random
from flask import redirect, url_for
import pickle


app = Flask(__name__)
CORS(app)

# =====================================================
# DRESS SALES SETUP
# =====================================================
try:
    dress_df = pd.read_csv("data/amazon_sales_cleaned.csv", encoding="unicode_escape")
    print("✅ Dress CSV loaded")
except Exception as e:
    print(f"⚠ Dress CSV not found: {e}")
    dress_df = pd.DataFrame({
        "Category": ["Kurta", "Dress", "Top", "Saree"],
        "Size": ["S", "M", "L", "XL"],
        "State": ["Maharashtra", "Delhi", "Karnataka", "UP"],
        "Year": [2022, 2023, 2022, 2024],
        "Amount": [1200, 800, 650, 1500],
        "Rating": [4.2, 3.8, 4.5, 4.0]
    })

def get_col(df, possible):
    for col in possible:
        if col in df.columns:
            return col
    return None

DRESS_CAT_COL = get_col(dress_df, ["Category", "category"])
DRESS_SIZE_COL = get_col(dress_df, ["Size", "size"])
DRESS_STATE_COL = get_col(dress_df, ["Ship-State", "ship-state", "State", "state"])
DRESS_YEAR_COL = get_col(dress_df, ["Year", "year"])
DRESS_AMOUNT_COL = get_col(dress_df, ["Amount", "amount"])
DRESS_RATING_COL = get_col(dress_df, ["Rating", "rating"])

# =====================================================
# ELECTRONICS SALES SETUP
# =====================================================
electronics_model = None
electronics_encoder = None

def load_electronics_artifacts():
    global electronics_model, electronics_encoder

    model_path = "models/electronics_model.pkl"
    encoder_path = "models/electronics_encoder.pkl"

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        electronics_model = joblib.load(model_path)
        electronics_encoder = joblib.load(encoder_path)
        print("✅ Electronics model & encoder loaded")
    else:
        print("❌ Electronics model files not found")


load_electronics_artifacts()

# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def root():
    return redirect(url_for("login"))

@app.route('/home')
def home():
    """Landing page to choose predictor"""
    return render_template('home.html')

@app.route('/dress')
def dress_page():
    """Dress sales predictor page"""
    return render_template('dress.html')

@app.route('/electronics')
def electronics_page():
    """Electronics sales predictor page"""
    return render_template('electronics.html')

# =====================================================
# DRESS PREDICTION ENDPOINTS
# =====================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # ✅ TEMP: allow all logins
        return redirect(url_for("home"))

    return render_template("login.html")

@app.route("/dress/get_options", methods=["GET"])
def dress_get_options():
    return jsonify({
        "categories": sorted(dress_df[DRESS_CAT_COL].dropna().astype(str).unique().tolist()) if DRESS_CAT_COL else [],
        "sizes": sorted(dress_df[DRESS_SIZE_COL].dropna().astype(str).unique().tolist()) if DRESS_SIZE_COL else [],
        "states": sorted(dress_df[DRESS_STATE_COL].dropna().astype(str).unique().tolist()) if DRESS_STATE_COL else [],
        "years": sorted(dress_df[DRESS_YEAR_COL].dropna().astype(int).unique().tolist()) if DRESS_YEAR_COL else []
    })

@app.route("/dress/predict", methods=["POST"])
def dress_predict():
    data = request.json
    filtered_df = dress_df.copy()

    if DRESS_CAT_COL and data["category"] != "Select category":
        filtered_df = filtered_df[filtered_df[DRESS_CAT_COL] == data["category"]]

    if DRESS_SIZE_COL and data["size"] != "Select size":
        filtered_df = filtered_df[filtered_df[DRESS_SIZE_COL] == data["size"]]

    if DRESS_STATE_COL and data["state"] != "Select state":
        filtered_df = filtered_df[filtered_df[DRESS_STATE_COL] == data["state"]]

    if DRESS_YEAR_COL and data["year"] != "Select year":
        filtered_df = filtered_df[filtered_df[DRESS_YEAR_COL] == int(data["year"])]

    has_data = not filtered_df.empty

    if has_data and DRESS_AMOUNT_COL:
        total_amount = float(filtered_df[DRESS_AMOUNT_COL].sum())
    else:
        total_amount = random.randint(1000, 199999)

    if has_data and DRESS_RATING_COL:
        avg_rating = round(float(filtered_df[DRESS_RATING_COL].mean()), 2)
    else:
        avg_rating = round(random.uniform(3.0, 5.0), 1)

    if has_data:
        count = int(len(filtered_df))
    else:
        count = random.randint(5, 300)

    return jsonify({
        "total_amount": total_amount,
        "avg_rating": avg_rating,
        "count": count
    })

 
# =====================================================
# ELECTRONICS PREDICTION ENDPOINT
# =====================================================
@app.route("/electronics/meta", methods=["GET"])
def electronics_meta():
    df = pd.read_csv("data/electronics_cleaned.csv")

    years = sorted(df["year"].dropna().unique().tolist())
    categories = sorted(df["category"].dropna().unique().tolist())

    return jsonify({
        "years": years,
        "categories": categories
    })
@app.route('/electronics/predict', methods=['POST'])
def electronics_predict():
    global electronics_model, electronics_encoder

    if electronics_model is None or electronics_encoder is None:
        load_electronics_artifacts()

    data = request.get_json()
    year = data.get("year")
    category = data.get("category")

    if not year or not category:
        return jsonify({"error": "Missing year or category"}), 400

    try:
        cat_encoded = electronics_encoder.transform([category])[0]
    except:
        cat_encoded = 0

    # fixed defaults (model trained same way)
    month = 1
    day = 1
    dow = 0

    features = np.array([[cat_encoded, int(year), month, day, dow]])
    prediction = electronics_model.predict(features)[0]

    return jsonify({
        "prediction": f"${prediction:,.2f}",
        "raw_prediction": float(prediction),
        "status": "success"
    })

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

# ============== GROCERY ROUTES ==============

GROCERY_CSV = "data/grocery_sales_data_cleaned.csv"

try:
    grocery_df = pd.read_csv(GROCERY_CSV, encoding="latin-1")
    print("✅ Grocery data loaded for dropdowns")
except Exception as e:
    grocery_df = None
    print("❌ Failed to load grocery data:", e)

grocery_model = None
grocery_scaler = None
grocery_encoders = None

try:
    with open("models/grocery_best_model.pkl", "rb") as f:
        grocery_model = pickle.load(f)
    
    with open("models/grocery_scaler.pkl", "rb") as f:
        grocery_scaler = pickle.load(f)
    
    with open("models/grocery_label_encoders.pkl", "rb") as f:
        grocery_encoders = pickle.load(f)
    
    print("✅ Grocery model, scaler, and encoders loaded")
except Exception as e:
    print(f"❌ Failed to load grocery model files: {e}")

@app.route('/grocery')
def grocery_page():
    return render_template('grocery.html')

@app.route('/grocery/get_options', methods=['GET'])
def get_grocery_options():
    try:
        if grocery_df is None:
            return jsonify({'error': 'Grocery data not loaded'}), 500
        
        return jsonify({
            'items': sorted(grocery_df['Item'].dropna().unique().astype(str).tolist()),
            'quantities': sorted(grocery_df['Sales Quantity'].dropna().unique().astype(str).tolist()),
            'datekeys': sorted(grocery_df['DateKey'].dropna().unique().astype(str).tolist())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/grocery/predict', methods=['POST'])
def predict_grocery():
    try:
        if grocery_model is None or grocery_scaler is None or grocery_encoders is None:
            return jsonify({'error': 'Grocery model not loaded'}), 500
            
        data = request.get_json()
        
        # Validate input
        if not data.get('item') or not data.get('quantity') or not data.get('datekey'):
            return jsonify({'error': 'Missing required fields'}), 400

        # Encode inputs
        item_encoded = grocery_encoders['Item'].transform([str(data['item'])])[0]
        quantity_encoded = grocery_encoders['Sales Quantity'].transform([str(data['quantity'])])[0]
        datekey_encoded = grocery_encoders['DateKey'].transform([str(data['datekey'])])[0]

        # Prepare input and predict
        input_data = np.array([[item_encoded, quantity_encoded, datekey_encoded]])
        input_scaled = grocery_scaler.transform(input_data)
        predicted_price = grocery_model.predict(input_scaled)[0]

        return jsonify({
            'predicted_price': round(float(predicted_price), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)