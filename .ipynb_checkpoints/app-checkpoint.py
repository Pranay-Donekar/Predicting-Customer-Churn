from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('best_gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Preprocess the data (assuming the preprocessing steps are encapsulated in a function)
    # df = preprocess_input(df)  # Implement this function based on your earlier preprocessing

    # Predict the probability of churn
    churn_prob = model.predict_proba(df)[:, 1][0]
    churn_pred = int(churn_prob >= 0.5)

    # Return the result as a JSON response
    return jsonify({'churn_probability': churn_prob, 'churn_prediction': churn_pred})

if __name__ == '__main__':
    app.run(debug=True)
