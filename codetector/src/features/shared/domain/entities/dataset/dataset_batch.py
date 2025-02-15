from dataclasses import dataclass

from ..samples import Sample


@dataclass(frozen=True)
class DatasetBatch(object):
    """
    Object containing a batch of samples returned by a dataset.
    """

    samples: list[Sample]
    """
    The list of samples returned in the batch.
    """

    final: bool
    """
    If it is the final batch (Whether all batches have been loaded).
    """