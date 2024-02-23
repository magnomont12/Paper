import numpy as np
import os
import pandas as pd

matrix = [
    ["image_grad-cam_scores_scores_basic_0_original_basic_basic", "image_grad-cam_scores_scores_basic_0_original_caco_caco", "image_grad-cam_scores_scores_basic_0_original_flat_flat", "image_grad-cam_scores_scores_basic_0_original_animated_animated"],
    ["image_grad-cam_scores_scores_caco_0_original_basic_basic", "image_grad-cam_scores_scores_caco_0_original_caco_caco", "image_grad-cam_scores_scores_caco_0_original_flat_flat", "image_grad-cam_scores_scores_caco_0_original_animated_animated"],
    ["image_grad-cam_scores_scores_flat_0_original_basic_basic", "image_grad-cam_scores_scores_flat_0_original_caco_caco", "image_grad-cam_scores_scores_flat_0_original_flat_flat", "image_grad-cam_scores_scores_flat_0_original_animated_animated"],
    ["image_grad-cam_scores_scores_animated_0_original_basic_basic", "image_grad-cam_scores_scores_animated_0_original_caco_caco", "image_grad-cam_scores_scores_animated_0_original_flat_flat", "image_grad-cam_scores_scores_animated_0_original_animated_animated"]
]

matrix_h = [[[] for b in range(len(matrix[a]))] for a in range(len(matrix))]
for a in range(len(matrix)):
    main_element =os.path.join("mean_images","grad-cam",f"{matrix[a][a]}.npy")
    main_element = np.load(main_element)
    main_element = (main_element - np.min(main_element)) / (np.max(main_element) - np.min(main_element))
    print(f"Rodada {a}", np.max(main_element))
    for b in range(len(matrix[a])):
        element = os.path.join("mean_images","grad-cam",f"{matrix[a][b]}.npy")
        element = np.load(element)
        element = (element - np.min(element)) / (np.max(element) - np.min(element))
        distance = np.linalg.norm(main_element - element)
        matrix_h[a][b] = distance

matrix_v = [[[] for b in range(len(matrix[a]))] for b in range(len(matrix[0]))]
for b in range(len(matrix[0])):
    main_element =os.path.join("mean_images","grad-cam",f"{matrix[b][b]}.npy")
    main_element = np.load(main_element)
    main_element = (main_element - np.min(main_element)) / (np.max(main_element) - np.min(main_element))
    for a in range(len(matrix[b])):
        print(matrix[b][b], matrix[a][b])
        element = os.path.join("mean_images","grad-cam",f"{matrix[a][b]}.npy")
        element = np.load(element)
        element = (element - np.min(element)) / (np.max(element) - np.min(element))
        distance = np.linalg.norm(main_element - element)
        matrix_v[a][b] = distance
matrix_h = pd.DataFrame(np.array(matrix_h)).to_csv("matrix_h.csv")
matrix_v = pd.DataFrame(np.array(matrix_v)).to_csv("matrix_v.csv")
