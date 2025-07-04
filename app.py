from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

app = Flask(__name__)

# Configuration
app.config.update(
    SECRET_KEY=os.urandom(24),
    MODEL_PATH=os.path.join('model', 'emission_model.joblib'),
    METADATA_PATH=os.path.join('model', 'model_metadata.joblib')
)

# Constants
EMISSION_THRESHOLDS = {
    'Low': 5000,
    'Moderate': 10000,
    'High': 15000
}
ENVIRONMENTAL_FACTORS = {
    'CO2_PER_CAR': 2.3,
    'CO2_PER_TREE': 0.06,
    'CO2_PER_HOUSEHOLD': 5.5
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.pipeline = None 
        self.metadata = None
        self.load_model()

    def load_model(self):
        try:
            self.pipeline = joblib.load(app.config['MODEL_PATH'])
            self.metadata = joblib.load(app.config['METADATA_PATH'])
            logger.info("Model and metadata loaded successfully.")
            logger.info(f"Pipeline steps: {[step[0] for step in self.pipeline.steps]}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data):
        try:
            expected_features = self.metadata['features']
            missing = set(expected_features) - set(input_data.keys())
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[expected_features]
            logger.info(f"Raw input for prediction:\n{input_df}")
            
            # Let the pipeline handle all transformations
            prediction = float(self.pipeline.predict(input_df)[0])
            logger.info(f"Raw prediction before clipping: {prediction}")
            
            return max(0, prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

class InputValidator:
    @staticmethod
    def validate_year(year):
        current_year = datetime.now().year
        try:
            year = int(year)
            if not (1990 <= year <= current_year + 5):
                raise ValueError(f"Year must be between 1990 and {current_year + 5}")
            return year
        except ValueError:
            raise ValueError("Invalid year format")

    @staticmethod
    def validate_numeric(value, field_name, min_value=0):
        try:
            value = float(value)
            if value < min_value:
                raise ValueError(f"{field_name} cannot be negative")
            return value
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}")

    @staticmethod
    def process_input(form_data):
        processed = {}
        
        # Validate and parse year
        processed['Year'] = InputValidator.validate_year(form_data.get('Year'))

        # Map form fields to model features
        field_mappings = {
            'Savanna_fires': 'Savanna fires',
            'Forest_fires': 'Forest fires',
            'Crop_Residues': 'Crop Residues',
            'Rice_Cultivation': 'Rice Cultivation',
            'Food_Transport': 'Food Transport',
            'Rural_population': 'Rural population',
            'Urban_population': 'Urban population'
        }

        for form_name, model_name in field_mappings.items():
            value = form_data.get(form_name) or form_data.get(model_name) or form_data.get(model_name.replace(' ', '_'))
            if value is None:
                raise ValueError(f"Missing field: {model_name}")
            processed[model_name] = InputValidator.validate_numeric(value, model_name)

        # Feature engineering (must exactly match training)
        processed['total_population'] = processed['Rural population'] + processed['Urban population']
        processed['urbanization_rate'] = processed['Urban population'] / processed['total_population'] if processed['total_population'] else 0
        processed['fire_contribution'] = processed['Savanna fires'] + processed['Forest fires']
        processed['agricultural_impact'] = processed['Crop Residues'] + processed['Rice Cultivation']
        
        # Ensure all expected features are present
        expected_features = [
            'Year', 'Savanna fires', 'Forest fires', 'Crop Residues',
            'Rice Cultivation', 'Food Transport', 'Rural population',
            'Urban population', 'total_population', 'urbanization_rate',
            'fire_contribution', 'agricultural_impact'
        ]
        
        missing = set(expected_features) - set(processed.keys())
        if missing:
            raise ValueError(f"Missing engineered features: {missing}")
            
        return processed

class ResultsAnalyzer:
    @staticmethod
    def get_emission_level(prediction):
        for level, threshold in EMISSION_THRESHOLDS.items():
            if prediction < threshold:
                return level
        return "Critical"

    @staticmethod
    def calculate_environmental_impact(prediction):
        return {
            'cars_equivalent': prediction / ENVIRONMENTAL_FACTORS['CO2_PER_CAR'],
            'trees_needed': prediction / ENVIRONMENTAL_FACTORS['CO2_PER_TREE'],
            'households_equivalent': prediction / ENVIRONMENTAL_FACTORS['CO2_PER_HOUSEHOLD']
        }

    @staticmethod
    def get_visualization_data(prediction):
        max_emission = max(EMISSION_THRESHOLDS.values())
        return {
            'gauge_angle': min((prediction / max_emission) * 180, 180),
            'percentage': min((prediction / max_emission) * 100, 100)
        }

# Load model and metadata
model_manager = ModelManager()

@app.route('/')
def home():
    try:
        return render_template('index.html', test_metrics=model_manager.metadata['metrics']['test'])
    except Exception as e:
        logger.error(f"Home route error: {e}")
        return f"<h1>Error</h1><p>Could not load home page: {e}</p>", 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = InputValidator.process_input(request.form)
            prediction = model_manager.predict(input_data)

            emission_level = ResultsAnalyzer.get_emission_level(prediction)
            environmental_impact = ResultsAnalyzer.calculate_environmental_impact(prediction)
            viz_data = ResultsAnalyzer.get_visualization_data(prediction)

            return render_template('results.html',
                                   prediction=round(prediction, 2),
                                   emission_level=emission_level,
                                   environmental_impact=environmental_impact,
                                   viz_data=viz_data,
                                   input_data=input_data)
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return render_template('predict.html',
                                   error=str(e),
                                   form_data=request.form,
                                   test_metrics=model_manager.metadata['metrics']['test'])
        except Exception as e:
            logger.error(f"Prediction route error: {e}")
            return f"<h1>Error</h1><p>{e}</p>", 500

    return render_template('predict.html', test_metrics=model_manager.metadata['metrics']['test'])

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return "<h1>Error 500</h1><p>An internal server error occurred.</p>", 500

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.error(f"Unhandled exception: {error}")
    return f"<h1>Unexpected Error</h1><p>{str(error)}</p>", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)