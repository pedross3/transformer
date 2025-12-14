# coding=utf-8
"""
Cross-View Transformer (CVT) encoder for DirectionNet.

Design goals:
- Drop-in replacement for SiameseEncoder in model.py
- Keep decoder interface the same: output [B, 1, 1, 1024] with attribute `inplanes = 1024`
- Flow: Patch embedding -> per-view self-attn -> bi-directional cross-attn (I0<->I1)
        -> average CLS tokens -> 512-D -> project to 1024 and reshape to 1x1

Note: Implemented with tf.compat.v1.keras to match the codebase.
"""
import math as m
import tensorflow.compat.v1 as tf # type: ignore
from tensorflow.compat.v1 import keras # type: ignore
from tensorflow.compat.v1.keras import regularizers # type: ignore
from tensorflow.compat.v1.keras.layers import Conv2D, Layer, Dense, Dropout, LayerNormalization # type: ignore


def gelu(x):
  """GELU activation (approximation) compatible with TF1 graph mode.

  Uses tanh approximation from Hendrycks & Gimpel (2016):
    0.5 * x * (1 + tanh(\sqrt(2/\pi) * (x + 0.044715 x^3)))
  """
  c0 = tf.constant(0.5, dtype=x.dtype)
  c1 = tf.constant(1.0, dtype=x.dtype)
  c2 = tf.constant(0.044715, dtype=x.dtype)
  sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)  # sqrt(2/pi)
  return c0 * x * (c1 + tf.tanh(sqrt_2_over_pi * (x + c2 * tf.pow(x, 3))))

def sinusoidal_positional_embedding(num_positions, dim):
  """
  Create sinusoidal positional embeddings.
  Returns: [num_positions, dim]
  """
  position = tf.cast(tf.range(num_positions)[:, tf.newaxis], tf.float32)
  div_term = tf.exp(
      tf.cast(tf.range(0, dim, 2), tf.float32) *
      -(m.log(10000.0) / dim)
  )

  pe = tf.zeros([num_positions, dim])
  pe_even = tf.sin(position * div_term)
  pe_odd = tf.cos(position * div_term)

  pe = tf.concat([pe_even, pe_odd], axis=1)
  pe = pe[:, :dim]  # safety slice
  return pe


class MultiHeadSelfAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
    super(MultiHeadSelfAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    # Dense() applies operator ((input) . W + b), we set use_bias=False
    self.qkv = Dense(3 * embed_dim, use_bias=False) # for Q, K, V together 
    self.proj = Dense(embed_dim, use_bias=False) # output projection 
    self.drop = Dropout(dropout_rate) # dropout layer 

  def call(self, x, training=False):
    # Self-attention implementation 
    # x: [B, T, C]
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = x.shape.as_list()[-1]
    qkv = self.qkv(x)  # [B, T, 3C]
    q, k, v = tf.split(qkv, 3, axis=-1)

    def reshape_heads(y):
      # y: [B, T, C] -> [B, H, T, D] where C = H*D 
      # B: batch size, T: sequence length, H: num_heads, D: head_dim
      # Reshape and transpose for multi-head attention
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])  # [B, H, T, D]

    # Reshape q, k, v for multi-head attention
    q = reshape_heads(q)
    k = reshape_heads(k)
    v = reshape_heads(v)

    # Compute attention scores
    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,T,T]
    attn = tf.nn.softmax(attn_logits, axis=-1) # Softmax over last axis (T)
    attn = self.drop(attn, training=training) # Apply dropout to attention weights
    # Compute attention output 
    # Transpose and reshape back to [B, T, C]
    out = tf.matmul(attn, v)  # [B,H,T,D]
    out = tf.transpose(out, [0, 2, 1, 3])  # [B,T,H,D]
    out = tf.reshape(out, [B, T, self.embed_dim])
    # Final linear projection
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MultiHeadCrossAttention(Layer):
  # Similar to MultiHeadSelfAttention but with separate Q and KV inputs
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
    super(MultiHeadCrossAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.q = Dense(embed_dim, use_bias=False)
    self.kv = Dense(2 * embed_dim, use_bias=False)
    self.proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

  def call(self, x_q, x_kv, training=False):
    # x_q: [B, Tq, C], x_kv: [B, Tk, C]
    B = tf.shape(x_q)[0]
    Tq = tf.shape(x_q)[1]
    Tk = tf.shape(x_kv)[1]
    C = x_q.shape.as_list()[-1]

    q = self.q(x_q)
    kv = self.kv(x_kv)
    k, v = tf.split(kv, 2, axis=-1)

    def reshape_heads(y, T):
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])  # [B,H,T,D]

    q = reshape_heads(q, Tq)
    k = reshape_heads(k, Tk)
    v = reshape_heads(v, Tk)

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,Tq,Tk]
    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v)  # [B,H,Tq,Tk]x[B,H,Tk,D]->[B,H,Tq,D]
    out = tf.transpose(out, [0, 2, 1, 3])  # [B,Tq,H,D]
    out = tf.reshape(out, [B, Tq, self.embed_dim])
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MLP(Layer):
  # Feed forward network with two Dense layers and GELU activation
  """
  Docstring for MLP
  Classic Vision Transformer FFN Block:
  Dense(embed_dim * mlp_ratio) -> GELU -> Dropout -> Dense(embed_dim) -> Dropout
  """
  def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(MLP, self).__init__(**kwargs)
    hidden = int(embed_dim * mlp_ratio)
    # Avoid tf.nn.gelu under tf.compat.v1; apply GELU manually in call().
    self.fc1 = Dense(hidden, activation=None)
    self.fc2 = Dense(embed_dim)
    self.drop = Dropout(dropout_rate)

  def call(self, x, training=False):
    x = self.fc1(x) # [B,T,hidden]
    x = gelu(x)
    x = self.drop(x, training=training) # dropout after GELU
    x = self.fc2(x) # [B,T,embed_dim]
    x = self.drop(x, training=training) # dropout after second Dense
    return x


class TransformerBlock(Layer):
  """
  Docstring for TransformerBlock
  Classic Vision Transformer Block:
  LayerNorm -> MultiHeadSelfAttention -> Add & Norm -> MLP -> Add
  """
  def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    # Norm is important before attention and MLP because of residual connections
    # x = x + MHSelfAttn(LayerNorm(x))
    # x = x + MLP(LayerNorm(x))
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.mlp = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, x, training=False):
    x = x + self.attn(self.norm1(x), training=training)
    x = x + self.mlp(self.norm2(x), training=training)
    return x


class CrossViewTransformerEncoder(keras.Model):
  """
  Encoder implementing Cross-View Transformer (CVT) for DirectionNet.
  Input: two images [B,H,W,3]
  Output: single tensor [B,1,1,1024] for decoder"""
  def __init__(self,
               embed_dim=512,
               num_layers=4,
               num_heads=8,
               mlp_ratio=4.0,
               patch_size=16,
               dropout_rate=0.0,
               regularization=0.01):
    super(CrossViewTransformerEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.inplanes = 1024  # to match decoder expectation

    # Patch embedding via conv
    # Converts image to patches and projects to embed_dim
    # lke ViT patch embedding
    # A 16x16 patch -> stride 16 -> produces ~(H/16)x(W/16) tokens

    self.patch_embed = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    # Positional embedding (added to tokens, incl. CLS)
    self.pos_embed = None  # lazily initialized based on input HxW
    # CLS token for classification 
    # A learnable token that summarizes each image
    # Every input gets the same CLS token added
    # Later, CLS tokens from both views are cross-attended
    # Shape: [1, 1, embed_dim]
    self.cls_token = self.add_weight(
        name='cls_token', shape=[1, 1, embed_dim],
        initializer=tf.random_normal_initializer(stddev=0.02))

    # Several layers of Transformer blocks for per-view processing
    # e.g 4 blocks, each doing self-attention + MLP
    self.blocks = [
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]
    # Cross attention for CLS tokens only (lightweight)
    # Why only CLS? 
    # - Efficiency: CLS tokens are small (1 token each)
    # - Avoids quadratic attention between patches of both images
    # - Encourages global-level fusion only
    self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)
    self.norm = LayerNormalization(epsilon=1e-6)
    # Final projection to 1024-D for decoder
    self.proj_to_decoder = Dense(1024)
    
    # add cls positional embedding
    self.cls_pos_embed = self.add_weight(
    name='cls_pos_embed',
    shape=[1, 1, embed_dim],
    initializer=tf.random_normal_initializer(stddev=0.02))


  def _flatten_patches(self, x):
    # x: [B,H,W,C] -> tokens [B, T, C]
    # Turns the 2D feature map into a sequence of tokens
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = x.shape.as_list()[-1]
    tokens = tf.reshape(x, [B, H * W, C])
    return tokens, H, W

  def _add_positional(self, tokens, H, W, training):
    """
    tokens: [B, T, C] where T = H*W
    """
    B = tf.shape(tokens)[0]
    T = tf.shape(tokens)[1]
    C = tokens.shape.as_list()[-1]

    # Create sinusoidal embedding for patches
    pos_emb = sinusoidal_positional_embedding(T, C)  # [T, C]
    pos_emb = tf.expand_dims(pos_emb, axis=0)        # [1, T, C]

    return tokens + pos_emb


  def call(self, img1, img2, training=False):
    # Forward pass of CVT encoder
    # Patch embedding
    # Each image is converted into patch tensors
    p1 = self.patch_embed(img1)  # [B, Hp, Wp, C]
    p2 = self.patch_embed(img2)
    t1, H1, W1 = self._flatten_patches(p1) # into sequences like
    #                                        [B, num_patches, embed_dim]
    t2, H2, W2 = self._flatten_patches(p2)

    # Add positional embeddings
    t1 = self._add_positional(t1, H1, W1, training)
    t2 = self._add_positional(t2, H2, W2, training)

    # Prepend CLS token to each
    B = tf.shape(t1)[0]
    # Add the same CLS token to each example in the batch
    cls = tf.tile(self.cls_token + self.cls_pos_embed, [B, 1, 1])
    x1 = tf.concat([cls, t1], axis=1)
    x2 = tf.concat([cls, t2], axis=1)
    """cls = tf.tile(self.cls_token, [B, 1, 1])  # [B,1,C]
    x1 = tf.concat([cls, t1], axis=1)
    x2 = tf.concat([cls, t2], axis=1)"""

    # Per-view self-attention blocks
    for blk in self.blocks:
      # Each image is processed independently
      # blk: TransformerBlock
      x1 = blk(x1, training=training)
      x2 = blk(x2, training=training)

    # Cross-attention on CLS tokens (bi-directional)
    cls1 = x1[:, :1, :]  # [B,1,C]
    cls2 = x2[:, :1, :]
    tok1 = x1[:, 1:, :]
    tok2 = x2[:, 1:, :]

    """
    CLS1 attends to tokens of image 2
    CLS2 attends to tokens of image 1
    This allows information exchange between views at a global level
    """

    cls1 = cls1 + self.cross_attn(cls1, tok2, training=training)
    cls2 = cls2 + self.cross_attn(cls2, tok1, training=training)

    # Average two CLS tokens to a single 512-D vector
    cls_avg = 0.5 * (cls1[:, 0, :] + cls2[:, 0, :])  # [B,C]
    cls_avg = self.norm(cls_avg)

    # Project to 1024 and reshape to 1x1 for the decoder
    # In order to match decoder input expectations
    y = self.proj_to_decoder(cls_avg)  # [B,1024]
    y = y[:, tf.newaxis, tf.newaxis, :]  # [B,1,1,1024]
    return y
