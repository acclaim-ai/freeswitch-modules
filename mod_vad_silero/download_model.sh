#!/bin/bash

# Download Silero VAD model
MODEL_DIR="/usr/local/share/silero_vad"
MODEL_PATH="$MODEL_DIR/silero_vad.onnx"

echo "Searching for Silero VAD model files..."

# Create directory if it doesn't exist
sudo mkdir -p "$MODEL_DIR"

# Remove existing file if it exists (in case it's corrupted)
if [ -f "$MODEL_PATH" ]; then
    echo "Removing existing file..."
    sudo rm "$MODEL_PATH"
fi

# Pin to a release tag (overridable) so this matches the freeswitch-image build
# and stays reproducible — avoid tracking the mutable `master` branch.
SILERO_VAD_VERSION="${SILERO_VAD_VERSION:-v5.1.2}"
MODEL_URLS=(
    "https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad.onnx"
    "https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad_16k_op15.onnx"
    "https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad_half.onnx"
)

for url in "${MODEL_URLS[@]}"; do
    echo "Trying: $url"
    if wget -O "$MODEL_PATH" "$url"; then
        echo "File downloaded successfully using: $url"
        echo "File size: $(ls -lh "$MODEL_PATH" | awk '{print $5}')"
        
        # Verify it's not an HTML file
        if file "$MODEL_PATH" | grep -q "HTML"; then
            echo "ERROR: Downloaded file is HTML, not model file. Download failed."
            sudo rm "$MODEL_PATH"
            continue
        fi
        
        echo "File type: $(file "$MODEL_PATH")"
        
        # Check if it looks like a valid model file
        if file "$MODEL_PATH" | grep -E "(ONNX|data|binary)" > /dev/null; then
            echo "Model file appears valid!"
            break
        else
            echo "Warning: File doesn't appear to be a valid model file"
            sudo rm "$MODEL_PATH"
            continue
        fi
    else
        echo "Failed to download from: $url"
    fi
done

# Check if we have a valid file
if [ ! -f "$MODEL_PATH" ]; then
    echo "All download attempts failed."
    echo ""
    echo "Please check the available files at:"
    echo "https://github.com/snakers4/silero-vad/tree/master/src/silero_vad/data"
    echo ""
    echo "You may need to download the model file manually and place it at: $MODEL_PATH"
    exit 1
fi

# Set proper permissions
sudo chmod 644 "$MODEL_PATH"

echo "Model ready for use with mod_vad_silero" 
