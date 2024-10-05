import yaml
from docetl.api import Pipeline


def run_pipeline_service(yaml_config: str):
    config = yaml.safe_load(yaml_config)
    pipeline = Pipeline.from_dict(config)
    cost = pipeline.run()

    output_path = config["pipeline"]["output"]["path"]

    return {
        "message": "Pipeline executed successfully",
        "cost": cost,
        "output_file": output_path,
    }
