import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go

# Load the simulated data
df = pd.read_csv('simulated_patient_data.csv')

# Features (input data)
X = df[['Age (years)', 'Weight (kg)', 'Height (cm)', 'BMI', 'BSA (m^2)',
        'Clearance (L/h)', 'Vd (L)', 'Time (hours)']].values

# Target variable (serum concentration)
y = df['Concentration (mg/L)'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model with dropout and additional tuning
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Predict on the test set
y_pred = model.predict(X_test)

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Patient Concentration Prediction"),
    
    # Input form for patient data
    html.Div([
        html.H3("Enter Patient Details:"),
        html.Label("Age (years):"),
        dcc.Input(id="age", type="number", value=40),
        html.Label("Weight (kg):"),
        dcc.Input(id="weight", type="number", value=70),
        html.Label("Height (cm):"),
        dcc.Input(id="height", type="number", value=175),
        html.Label("BMI:"),
        dcc.Input(id="bmi", type="number", value=22.9),
        html.Label("BSA (m^2):"),
        dcc.Input(id="bsa", type="number", value=1.9),
        html.Label("Clearance (L/h):"),
        dcc.Input(id="cl", type="number", value=15.0),
        html.Label("Vd (L):"),
        dcc.Input(id="vd", type="number", value=100),
        html.Label("Time (hours):"),
        dcc.Input(id="time", type="number", value=5.0),
        html.Button("Predict", id="predict-btn", n_clicks=0),
    ], style={"margin-bottom": "20px"}),

    # Output for prediction
    html.Div(id="prediction-output", style={"margin-top": "20px"}),

    # Graph of Actual vs Predicted
    html.Div([
        html.H3("Actual vs Predicted Concentration"),
        dcc.Graph(id="actual-vs-predicted")
    ])
])

# Callback to handle prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("weight", "value"),
    State("height", "value"),
    State("bmi", "value"),
    State("bsa", "value"),
    State("cl", "value"),
    State("vd", "value"),
    State("time", "value")
)
def predict_concentration(n_clicks, age, weight, height, bmi, bsa, cl, vd, time):
    if n_clicks > 0:
        patient_data = [age, weight, height, bmi, bsa, cl, vd, time]
        patient_data_scaled = scaler.transform([patient_data])
        predicted_concentration = model.predict(patient_data_scaled)[0][0]
        return f"Predicted Concentration at {time} hours: {predicted_concentration:.2f} mg/L"
    return "Enter patient details and click Predict."

# Callback to update the Actual vs Predicted graph
@app.callback(
    Output("actual-vs-predicted", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_graph(n_clicks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred.flatten(), mode="markers", name="Predictions"))
    fig.add_trace(go.Line(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                          name="Ideal Line", line=dict(color="red", dash="dash")))
    fig.update_layout(
        title="Actual vs Predicted Concentration",
        xaxis_title="Actual Concentration",
        yaxis_title="Predicted Concentration",
        legend=dict(x=0, y=1)
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
