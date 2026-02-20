from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    # Run the pipeline with the path to the data
    print(f"Tracking URI: {Client().active_stack.experiment_tracker.get_tracking_uri()}")
    train_pipeline(data_path="/Users/george/Desktop/MachineLearning/MLOPS/data/olist_customers_dataset.csv")
        
