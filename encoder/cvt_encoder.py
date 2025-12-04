# coding utf-8

"""
Docstring for encoder.cvt_encoder

Cross-View Transformer (CvT) encoder implementation for DirectionNet.

Design Goals:
    - Replace Siamese Encoder with CvT Encoder.
    - Keep decoder interface the same: output shape (B, 1, 1, 1024).
    - Flow: Patch embedding -> Transformer Blocks (self-attention) -> Bi-directional cross-attention
            -> average CLS tokens -> 512-dim output projection -> final output (B, 1, 1, 1024) and reshape to 1x1.
"""
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Conv2D, Layer, Dense, Dropout, LayerNormalization




