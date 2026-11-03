#!/bin/bash
# Kaggle Execution Script for WaterSIC Quantization of Huihui-Ornith-1.5-35B-A3B
# Target Hardware: 2x NVIDIA T4 (16GB VRAM each), 30GB CPU RAM

set -e

echo "=========================================================================="
echo " Starting WaterSIC Quantization Pipeline on Kaggle (2x T4 GPU)"
echo " Model: huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated"
echo "=========================================================================="

# 1. Environment and Memory Optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_DISABLE_FILE_LOCKING=1
export TOKENIZERS_PARALLELISM=false

# 2. Install dependencies
echo "[1/4] Installing necessary dependencies..."
pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate safetensors datasets xxhash huggingface_hub

# 3. Check GPU status
echo "[2/4] Verifying GPU resources..."
nvidia-smi

# 4. Define paths
MODEL_ID="huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated"
PROFILE_PATH="profiles/qwen35_moe_35b.json"
OUTPUT_DIR="/kaggle/working/Huihui-Ornith-1.5-35B-A3B-WaterSIC"
OUTPUT_WSIC="${OUTPUT_DIR}/Huihui-Ornith-1.5-35B-A3B.wsic"

mkdir -p "${OUTPUT_DIR}"

# 5. Run Progressive Layer-by-Layer Quantization
echo "[3/4] Launching layer-by-layer progressive quantization..."
python quantize.py \
    --model_id "${MODEL_ID}" \
    --profile "${PROFILE_PATH}" \
    --output_wsic "${OUTPUT_WSIC}" \
    --nsamples 64 \
    --seqlen 2048

# 6. Verify Output
echo "[4/4] Verifying output container..."
ls -lh "${OUTPUT_WSIC}"

echo "=========================================================================="
echo " WaterSIC Quantization Finished Successfully!"
echo " Container saved at: ${OUTPUT_WSIC}"
echo "=========================================================================="
