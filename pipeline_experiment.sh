#!/bin/bash
suffix_list=('animated' 'basic' 'caco' 'flat')
type_viz=$1

for suffix in "${suffix_list[@]}"; do
    input_file="$type_viz/lista_de_pastas_$suffix.txt"

    create_gradients="scripts/create_gradients.py"

    create_visu="scripts/create_visu.py"

    create_video="scripts/create_video.py"

    create_mean="scripts/calculate_mean.py"

    if [ ! -f "$input_file" ]; then
        echo "Arquivo $input_file não encontrado."
        exit 1
    fi

    while IFS=, read -r model scenario image score blend; do
        echo "Model: $model | Scenario: $scenario | Image: $image | Score: $score | Blend: $blend"

        python3 "$create_gradients" "$model" "$image" "$score" "$type_viz"

        python3 "$create_visu" "$image" "$score" "$blend" "$type_viz"

        python3 "$create_video" "$blend"

        python3 "$create_mean" "$score" "$type_viz"
    done < "$input_file"
  
done