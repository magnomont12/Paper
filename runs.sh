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
  local folder_scores="${file_without_extension/scenarios/scores_$suffix}"
  local folder_blend="${file_without_extension/scenarios/blend_$suffix}"
  
  mkdir -p "$folder_image"
  mkdir -p "$folder_scores"
  mkdir -p "$folder_blend"

  echo "$path_model,$cfg_file,$folder_image,$folder_scores,$folder_blend" >> "$output_file"
  
}

suffix_list=('animated' 'basic' 'caco' 'flat')

for suffix in "${suffix_list[@]}"; do
  path_model="models/$suffix"
  output_file="lista_de_pastas_$suffix.txt"
  find_cfg_files "scenarios" "$suffix"
done