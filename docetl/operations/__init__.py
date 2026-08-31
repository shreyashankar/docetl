import importlib.metadata
from docetl.operations.cluster import ClusterOperation
from docetl.operations.code_operations import (
    CodeFilterOperation,
    CodeMapOperation,
    CodeReduceOperation,
)
from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.filter import FilterOperation
from docetl.operations.gather import GatherOperation
from docetl.operations.map import MapOperation, ParallelMapOperation
from docetl.operations.link_resolve import LinkResolveOperation
from docetl.operations.reduce import ReduceOperation
from docetl.operations.resolve import ResolveOperation
from docetl.operations.rank import RankOperation
from docetl.operations.split import SplitOperation
from docetl.operations.sample import SampleOperation
from docetl.operations.topk import TopKOperation
from docetl.operations.unnest import UnnestOperation
from docetl.operations.unnest_columns import UnnestColumnsOperation
from docetl.operations.scan import ScanOperation
from docetl.operations.add_uuid import AddUuidOperation
from docetl.operations.stipple_verify import StippleVerifyOperation
from docetl.operations.extract import ExtractOperation
from docetl.operations.web_fetch import WebFetchOperation
from docetl.operations.web_search import WebSearchOperation

mapping = {
    "cluster": ClusterOperation,
    "code_filter": CodeFilterOperation,
    "code_map": CodeMapOperation,
    "code_reduce": CodeReduceOperation,
    "equijoin": EquijoinOperation,
    "filter": FilterOperation,
    "gather": GatherOperation,
    "link_resolve": LinkResolveOperation,
    "map": MapOperation,
    "parallel_map": ParallelMapOperation,
    "reduce": ReduceOperation,
    "resolve": ResolveOperation,
    "rank": RankOperation,
    "split": SplitOperation,
    "sample": SampleOperation,
    "topk": TopKOperation,
    "unnest": UnnestOperation,
    "unnest_columns": UnnestColumnsOperation,
    "scan": ScanOperation,
    "add_uuid": AddUuidOperation,
    "stipple_verify": StippleVerifyOperation,
    "extract": ExtractOperation,
    "web_fetch": WebFetchOperation,
    "web_search": WebSearchOperation,
}


def get_operation(operation_type: str):
    """Loads a single operation by name"""
    try:
        entrypoint = importlib.metadata.entry_points(group="docetl.operation")[
            operation_type
        ]
        return entrypoint.load()
    except KeyError:
        if operation_type in mapping:
            return mapping[operation_type]
        raise KeyError(f"Unrecognized operation {operation_type}")


def get_operations():
    """Load all available operations and return them as a dictionary"""
    operations = mapping.copy()
    operations.update(
        {
            op.name: op.load()
            for op in importlib.metadata.entry_points(group="docetl.operation")
        }
    )
    return operations
