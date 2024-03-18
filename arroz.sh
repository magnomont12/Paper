#!/bin/bash
suffix_list=('animated' 'basic' 'caco' 'flat')
type_viz=$1

for suffix in "${suffix_list[@]}"; do
    input_file="$type_viz/lista_de_pastas_$suffix.txt"

    create_mean="scripts/calc_mean_step.py"

    if [ ! -f "$input_file" ]; then
        echo "Arquivo $input_file não encontrado."
        exit 1
    fi

    while IFS=, read -r model scenario image score blend score_normal; do
        echo "Model: $model | Scenario: $scenario | Image: $image | Score: $score | Blend: $blend"
        python3 "$create_mean" "$score_normal" "$type_viz"
    done < "$input_file"
done