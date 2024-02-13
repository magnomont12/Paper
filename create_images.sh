#!/bin/bash

find_cfg_files() {
  local folder=$1
  local suffix=$2

  for item in "$folder"/*; do
    if [ -d "$item" ]; then
      find_cfg_files "$item" "$suffix"
    elif [ -f "$item" ] && [[ "$item" == *.cfg ]]; then
      create_new_folder "$item" "$suffix"
    fi
  done
}

create_new_folder() {
  local cfg_file=$1
  local suffix=$2
  local file_without_extension="${cfg_file%.cfg}"
  local folder_image="${file_without_extension/scenarios/images_$suffix}"
  
  mkdir -p "images/$folder_image"

  echo "$path_model,$cfg_file,images/$folder_image" >> "images/$output_file"
  
}

type_viz=$1
echo "Type of visualization: $type_viz"

suffix_list=('animated' 'basic' 'caco' 'flat')

for suffix in "${suffix_list[@]}"; do
  path_model="models/$suffix"
  output_file="lista_de_pastas_$suffix.txt"
  find_cfg_files "scenarios" "$suffix"
done

suffix_list=('animated' 'basic' 'caco' 'flat')

for suffix in "${suffix_list[@]}"; do
    input_file="images/lista_de_pastas_$suffix.txt"
    
    python_script="scripts/run_agent.py"

    if [ ! -f "$input_file" ]; then
        echo "Arquivo $input_file não encontrado."
        exit 1
    fi

    while IFS=, read -r model scenario image; do
        echo "Model: $model | Scenario: $scenario | Image: $image"
        python3 "$python_script" "$model" "$scenario" "$image"
        break
    done < "$input_file"
    break
  
done
