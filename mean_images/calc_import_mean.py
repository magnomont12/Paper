import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cv2

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
    print(main_element.shape)
    #main_element = (main_element - np.min(main_element)) / (np.max(main_element) - np.min(main_element))
    main_element = main_element.ravel()
    print(main_element.shape)
    main_element = main_element.reshape(1,-1)
    print(f"Rodada {a}", np.max(main_element))
    for b in range(len(matrix[a])):
        element = os.path.join("mean_images","grad-cam",f"{matrix[a][b]}.npy")
        element = np.load(element)
        # element = (element - np.min(element)) / (np.max(element) - np.min(element))
        # distance = np.linalg.norm(main_element - element)
        element = element.ravel()
        element = element.reshape(1,-1)
        distance = cosine_similarity(main_element, element)[0][0]
        matrix_h[a][b] = distance

matrix_v = [[[] for b in range(len(matrix[a]))] for a in range(len(matrix[0]))]
for b in range(len(matrix[0])):
    main_element = os.path.join("mean_images","grad-cam",f"{matrix[b][b]}.png")
    main_element = cv2.imread(main_element)
    main_element = cv2.cvtColor(main_element, cv2.COLOR_BGR2GRAY)
    hist_main_element = cv2.calcHist([main_element], [0], None, [256], [0, 256])
    cv2.normalize(hist_main_element, hist_main_element, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    for a in range(len(matrix[b])):
        print(matrix[b][b], matrix[a][b])
        element = os.path.join("mean_images","grad-cam",f"{matrix[a][b]}.png")
        element = cv2.imread(element)
        element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)
        hist_element = cv2.calcHist([element], [0], None, [256], [0, 256])
        cv2.normalize(hist_element, hist_element, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        distance = resultado = cv2.compareHist(hist_main_element, hist_element, cv2.HISTCMP_BHATTACHARYYA)
        matrix_h[b][a] = distance
        print(distance)
        
matrix_h = pd.DataFrame(np.array(matrix_h)).to_csv("matrix_h.csv")
matrix_v = pd.DataFrame(np.array(matrix_v)).to_csv("matrix_v.csv")
