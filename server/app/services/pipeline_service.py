from docetl.runner import DSLRunner


def run_pipeline_service(yaml_config: str):
    try:
        runner = DSLRunner.from_yaml(yaml_config)
        cost = runner.run()

        return {
            "message": "Pipeline executed successfully",
            "cost": cost,
        }
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
