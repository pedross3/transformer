# database is here: https://drive.google.com/file/d/1SB1g8ectHyhG23g8FqiwKJApyvmK5Sc9/view?usp=sharing
# paper is here https://arxiv.org/pdf/2203.10694

"""
FO and FA mimics the functionality of the original Transformer architecture as closely as possible while being
optimized for fast autoregressive inference. Which means, that the object detection + self-attention is done, 
but fully on the frequency domain using Fourier transforms, enabling the FAR model to be robust and efficient 
to moving backgrounds or dynamic cameras.

The FO (Fourier Object Disentanglement) separates the object from the background withput needing any bounding box
FA captures long range space-time relationships at a very lower computational cost using FFT 
"""

"""
1- load images from databse
2- extract features C T W H - Channel, Time, Width and Height
3- pass through FO and FA transformer blocks^
4- 
"""