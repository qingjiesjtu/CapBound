for model_name in \
  "gpt-oss-20b" \
  "DeepSeek-R1-Distill-Qwen-32B" \
  "DeepSeek-R1-0528-Qwen3-8B" \
  "QwQ-32B"
do
    echo "Running with model: $model_name"
    python capabilityBoundary.py --modelname "$model_name"
done

wait 
echo "All extraction processess completed."
