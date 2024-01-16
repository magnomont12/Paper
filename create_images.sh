#!/bin/bash
suffix_list=('animated' 'basic' 'caco' 'flat')

for suffix in "${suffix_list[@]}"; do
    input_file="lista_de_pastas_$suffix.txt"
    
    python_script="scripts/run_agent.py"

    create_gradients="scripts/create_gradients.py"

    create_visu="scripts/create_visu.py"

    create_video="scripts/create_video.py"

    if [ ! -f "$input_file" ]; then
        echo "Arquivo $input_file não encontrado."
        exit 1
    fi

    while IFS=, read -r model scenario image score blend; do
        echo "Model: $model | Scenario: $scenario | Image: $image | Score: $score | Blend: $blend"
        python3 "$python_script" "$model" "$scenario" "$image"

        python3 "$create_gradients" "$model" "$image" "$score"

        python3 "$create_visu" "$image" "$score" "$blend"

        python3 "$create_video" "$blend"
        break
    done < "$input_file"
  
done