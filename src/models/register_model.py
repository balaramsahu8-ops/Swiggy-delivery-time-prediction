import mlflow
import dagshub
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='balaram.sahu8', repo_name='Swiggy-delivery-time-prediction', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/balaram.sahu8/Swiggy-delivery-time-prediction.mlflow")


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    model_name = "delivery_time_pred_model"
    client = MlflowClient()
    experiment = client.get_experiment_by_name("DVC Pipeline")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished DVC Pipeline MLflow run is available")

    run = runs[0]
    run_id = run.info.run_id
    artifact_source = f"{run.info.artifact_uri}/{model_name}"
    model_version = client.create_model_version(
        name=model_name,
        source=artifact_source,
        run_id=run_id,
    )
    
    
    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
    