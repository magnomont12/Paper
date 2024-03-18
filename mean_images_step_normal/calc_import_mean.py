import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def calc_hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist /= hist.sum()
    return hist

def bhattacharyya_distance(hist1, hist2):
    bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
    distance = -np.log(bhattacharyya_coeff)
    return distance

def calc_pearson_correlation(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    
    flat_image1 = image1.flatten()
    flat_image2 = image2.flatten()
    
    mean_image1 = np.mean(flat_image1)
    mean_image2 = np.mean(flat_image2)
    
    numerator = np.sum((flat_image1 - mean_image1) * (flat_image2 - mean_image2))
    denominator = np.sqrt(np.sum((flat_image1 - mean_image1)**2) * np.sum((flat_image2 - mean_image2)**2))
    pearson_correlation = numerator / denominator
    
    return pearson_correlation

def calc_distances_original(step):
    matrix = [
        ["image_grad-cam_scores_normal_scores_basic_0_original_basic_basic", "image_grad-cam_scores_normal_scores_basic_0_original_caco_caco", "image_grad-cam_scores_normal_scores_basic_0_original_flat_flat", "image_grad-cam_scores_normal_scores_basic_0_original_animated_animated"],
        ["image_grad-cam_scores_normal_scores_caco_0_original_basic_basic", "image_grad-cam_scores_normal_scores_caco_0_original_caco_caco", "image_grad-cam_scores_normal_scores_caco_0_original_flat_flat", "image_grad-cam_scores_normal_scores_caco_0_original_animated_animated"],
        ["image_grad-cam_scores_normal_scores_flat_0_original_basic_basic", "image_grad-cam_scores_normal_scores_flat_0_original_caco_caco", "image_grad-cam_scores_normal_scores_flat_0_original_flat_flat", "image_grad-cam_scores_normal_scores_flat_0_original_animated_animated"],
        ["image_grad-cam_scores_normal_scores_animated_0_original_basic_basic", "image_grad-cam_scores_normal_scores_animated_0_original_caco_caco", "image_grad-cam_scores_normal_scores_animated_0_original_flat_flat", "image_grad-cam_scores_normal_scores_animated_0_original_animated_animated"]
    ]
    matrix_h = [[[] for b in range(len(matrix[a]))] for a in range(len(matrix))]
    for a in range(len(matrix)):
        main_element =os.path.join("mean_images_step_normal","grad-cam",step, f"{matrix[a][a]}.png")
        main_element = cv2.imread(main_element)
        for b in range(len(matrix[a])):
            element = os.path.join("mean_images_step_normal","grad-cam",step, f"{matrix[a][b]}.png")
            element = cv2.imread(element)
            distance = calc_pearson_correlation(main_element, element)
            matrix_h[a][b] = distance

    matrix_v = [[[] for b in range(len(matrix[a]))] for a in range(len(matrix[0]))]
    for b in range(len(matrix[0])):
        main_element = os.path.join("mean_images_step_normal","grad-cam",step,f"{matrix[b][b]}.png")
        main_element = cv2.imread(main_element)
        for a in range(len(matrix[b])):
            element = os.path.join("mean_images_step_normal","grad-cam", step, f"{matrix[a][b]}.png")
            element = cv2.imread(element)
            distance = calc_pearson_correlation(main_element, element)
            matrix_v[b][a] = distance
            
    matrix_h = pd.DataFrame(np.array(matrix_h)).to_csv(f"mean_images_step_normal/matrix_h_{step}.csv")
    matrix_v = pd.DataFrame(np.array(matrix_v)).to_csv(f"mean_images_step_normal/matrix_v_{step}.csv")

def calc_distance_blends(step):
    models = ["basic", "caco", "flat", "animated"]
    for model in models:
        matrix_blend = [
            [f"image_grad-cam_scores_normal_scores_{model}_0_original_basic_basic", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_animated_ba_25", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_animated_ba_50", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_animated_ba_75", f"image_grad-cam_scores_normal_scores_{model}_0_original_animated_animated"],
            [f"image_grad-cam_scores_normal_scores_{model}_0_original_basic_basic", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_caco_bc_25", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_caco_bc_50", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_caco_bc_75", f"image_grad-cam_scores_normal_scores_{model}_0_original_caco_caco"],
            [f"image_grad-cam_scores_normal_scores_{model}_0_original_basic_basic", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_flat_bf_25", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_flat_bf_50_1", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_flat_bf_50_2", f"image_grad-cam_scores_normal_scores_{model}_2_blend_basic_flat_bf_75", f"image_grad-cam_scores_normal_scores_{model}_0_original_flat_flat"]
        ]
        matrix_h = [[[] for b in range(len(matrix_blend[a]))] for a in range(len(matrix_blend))]
        for a in range(len(matrix_blend)):
            main_element =os.path.join("mean_images_step_normal","grad-cam",step,f"{matrix_blend[a][0]}.png")
            main_element = cv2.imread(main_element)
            for b in range(len(matrix_blend[a])):
                element = os.path.join("mean_images_step_normal","grad-cam",step,f"{matrix_blend[a][b]}.png")
                element = cv2.imread(element)
                distance = calc_pearson_correlation(main_element, element)
                matrix_h[a][b] = distance
        matrix_h[0].append(None)
        matrix_h[1].append(None)
        matrix_h = pd.DataFrame(np.array(matrix_h)).to_csv(f"mean_images_step_normal/matrix_h_{step}_{model}.csv")

if __name__ == "__main__":
    for step in range(1,11):
        calc_distances_original(str(step))
        calc_distance_blends(str(step))