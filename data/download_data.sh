#!/usr/bin/env bash

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URLS=(
    "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
    "https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist"
    "https://www.kaggle.com/api/v1/datasets/download/awsaf49/clean-weather-dataset"
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

for URL in "${URLS[@]}"; do
    FILE_NAME=$(basename "$URL")

    if [[ ! "$FILE_NAME" =~ \.(txt|zip)$ ]]; then
        FILE_NAME="${FILE_NAME}.zip"
    fi

    OUTPUT_PATH="${DATA_DIR}/${FILE_NAME}"

    echo "Downloading $URL -> $OUTPUT_PATH ..."
    curl -L -o "$OUTPUT_PATH" "$URL"
    echo "✅ Done."
    echo
done
