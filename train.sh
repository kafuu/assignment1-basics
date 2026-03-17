uv run ./cs336_basics/train.py \
    --train_data_path ./data/train.bin \
    --valid_data_path ./data/valid.bin \
    --run_name "baseline_model" \
    --batch_size 64 \
    --context_length 512 \
    --lr 6e-4 \