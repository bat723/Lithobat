"""
litho_sim.ml
============
Machine-learning components for wafer-image anomaly detection.

Provides a convolutional autoencoder trained on clean aerial images.
At inference, images with reconstruction errors above a calibrated
threshold are flagged as defective.

Submodules
----------
dataset   – PyTorch Dataset / DataLoader helpers
models    – ConvAutoencoder and SimpleAutoencoder architectures
train     – Training loop, evaluation, and anomaly scoring utilities
"""

