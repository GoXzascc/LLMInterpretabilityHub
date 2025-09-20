for model_name in "EleutherAI/pythia-70m"; do
    if [ $model_name == "EleutherAI/pythia-70m" ]; then
        k=51
        model_layers=(3 4)
    elif [ $model_name == "google/gemma-2-2b" ]; then
        k=230
        model_layers=(12 16)
    elif [ $model_name == "Qwen/Qwen3-4B-Thinking-2507" ]; then
        k=256
        model_layers=(18 20)
    elif [ $model_name == "openai-community/gpt2" ]; then
        k=76
        model_layers=(6 8)
    fi

    for model_layer in ${model_layers[@]}; do
        uv run src/trainer.py \
        --model_name $model_name \
        --layer $model_layer \
        --dataset "monology/pile-uncopyrighted" \
        --batch_size 128 \
        --k $k \
        --sae_name "batchtopk" \
        --training_steps 30000
    done
done