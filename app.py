import os
import sys
from flask import Flask, request, render_template

from source.Pipeline.predict_pipeline import CustomData, PredictPipeline
from source.Pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)
app = application


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for model prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        # Construct CustomData object matching HTML form names and types
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print("Dataframe generated for prediction:\n", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])


@app.route('/train', methods=['GET', 'POST'])
def train_pipeline_route():
    try:
        pipeline = TrainPipeline()
        r2_score = pipeline.run_pipeline()
        return f"Training Completed Successfully! Final Model R2 Score: {r2_score:.4f}"
    except Exception as e:
        return f"Error occurred during training: {str(e)}"


if __name__ == "__main__":
    # Uses port 8000 to prevent port conflicts on macOS
    app.run(host="0.0.0.0", port=8000, debug=True)