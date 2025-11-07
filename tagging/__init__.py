"""
Tagging package for ASTE project.

Contains span tagging schemes and inference algorithms.
"""

from tagging.span_tagging import (
    create_tag_table,
    create_label_maps,
    create_sentiment_maps,
    convert_tags_to_ids,
    convert_ids_to_tags
)
from tagging.inference import extract_triplets_from_tags

__all__ = [
    'create_tag_table',
    'create_label_maps',
    'create_sentiment_maps',
    'convert_tags_to_ids',
    'convert_ids_to_tags',
    'extract_triplets_from_tags'
]
