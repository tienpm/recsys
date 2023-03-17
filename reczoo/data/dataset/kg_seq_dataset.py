# -*- coding: utf-8 -*-
r"""
reczoo.data.kg_seq_dataset
#############################
"""

from reczoo.data.dataset import SequentialDataset, KnowledgeBasedDataset


class KGSeqDataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~reczoo.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~reczoo.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """

    def __init__(self, config):
        super().__init__(config)