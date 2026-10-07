"""
modality_bias.mb
================
Core modules for modality bias detection and mitigation.

Modules
-------
cremad_data     : CREMA-D dataset loader (audio spectrogram + video frames)
cmu_mosei_data  : CMU-MOSEI dataset loader (audio + video + text)
ig_xai_av       : Integrated Gradients attribution for audio+video modalities
mbs             : Modality Bias Score computation
correlation     : Pearson + Spearman correlation between XAI scores and MBS
ablation        : Modality ablation and confidence drop analysis
layerwise       : Per-layer bias analysis
"""
