import sys
from source.exception import CustomException
from source.logger import logging
from source.components.data_ingestion import DataIngestion
from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline...")

            # 1. Initiate Data Ingestion
            logging.info("Initiating Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # 2. Initiate Data Transformation
            logging.info("Initiating Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # 3. Initiate Model Trainer
            logging.info("Initiating Model Trainer")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed successfully. Model R2 Score: {r2_score}")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        r2_score = pipeline.run_pipeline()
        print(f"Training successfully finished! Final Model R2 Score: {r2_score:.4f}")
    except Exception as e:
        print(f"Pipeline Execution Failed: {str(e)}")