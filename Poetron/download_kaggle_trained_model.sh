#!/bin/bash
# Download the trained Flavourtown Poetry Generator model from Kaggle

echo "Downloading trained model from Kaggle..."
curl -L -o ~/Downloads/flavourtownpoetrongeneratormodel.zip \
  https://www.kaggle.com/api/v1/datasets/download/xongkoro/flavourtownpoetrongeneratormodel

if [ $? -eq 0 ]; then
    echo "✓ Download complete"
    echo "Extracting model..."
    
    # Create models directory if it doesn't exist
    mkdir -p models/kaggle_trained_model
    
    # Extract to models directory
    unzip -q ~/Downloads/flavourtownpoetrongeneratormodel.zip -d models/kaggle_trained_model
    
    echo "✓ Model extracted to models/kaggle_trained_model/"
    ls -lah models/kaggle_trained_model/
else
    echo "Download failed. Check your internet connection and Kaggle dataset URL."
    exit 1
fi
