import os
from kedro.framework.hooks import hook_impl
import logging
from kedro.io import MemoryDataset

logger = logging.getLogger(__name__)

class CleanupDataHook:
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        pipeline_name = run_params.get("pipeline_name")

        if pipeline_name is None:
            data_folder = os.path.abspath("data")

            if os.path.exists(data_folder):
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                        except Exception as E:
                            logger.error(E)

                logger.info('Cleared all the intermediate outputs')

            else:
                logger.info("Data folder does not exist")

        else:
            logger.info(f"Skipping data folder clean up since pipeline {pipeline_name} is specified")    
