#!/bin/bash
# Rename model files to remove + characters
cd "$(dirname "$0")"
mv "BiLSTM+Attention.pt" "BiLSTM_Attention.pt" 2>/dev/null || echo "BiLSTM_Attention.pt already exists"
mv "BiLSTM+CNN.pt" "BiLSTM_CNN.pt" 2>/dev/null || echo "BiLSTM_CNN.pt already exists"
mv "BiLSTM+CNN+Attention.pt" "BiLSTM_CNN_Attention.pt" 2>/dev/null || echo "BiLSTM_CNN_Attention.pt already exists"
echo "Model files renamed successfully!"
