"""
Models package for ASTE project.

Contains neural network model architectures.
"""

from models.aste_model import ASTEModel

# Backward compatibility for older saved models
base_model = ASTEModel

__all__ = ['ASTEModel', 'base_model']
