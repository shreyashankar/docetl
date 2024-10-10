from jinja2 import Environment, Template
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base import BaseOperation
from .utils import RichLoopBar
from .clustering_utils import get_embeddings_for_clustering

class OutliersOperation(BaseOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_batch_size: int = self.config.get(
            "max_batch_size", kwargs.get("max_batch_size", float("inf"))
        )

    def syntax_check(self) -> None:
        """
        Checks the configuration of the OutlierOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing
        """

        pass

    
    def execute(
        self, input_data: List[Dict], is_build: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Executes the cluster operation on the input data. Modifies the
        input data and returns it in place.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the filtered
              list of dictionaries and the total cost of the operation.
        """
        
        embeddings, cost = get_embeddings_for_clustering(
            input_data, self.config, self.runner.api
        )

        embeddings = np.array(embeddings)
        center = embeddings.mean(axis=0)
        
        distances = np.sqrt(((embeddings - center)**2).sum(axis=1))

        if "percentile" in self.config:
            distance_distribution = np.sort(distances)
            cutoff = distance_distribution[int(self.config["percentile"] / 100. * (len(distance_distribution)-1))]
        elif "std" in self.config:
            cutoff = np.sqrt((embeddings.std(axis=0)**2).sum()) * self.config["std"]
        
        include = distances <= cutoff
            
        return [
            item
            for idx, item in enumerate(input_data)
            if include[idx]], cost
        
