import os
import cv2
import numpy as np

def extrair_valores(string):
    """
    Extracts x and y values from a string.

    Args:
        string (str): The input string in the format "x_y".

    Returns:
        tuple: A tuple containing the extracted x and y values.

    """
    splits = string.split("_")
    x = int(splits[1])
    y = int(splits[2])
    return x, y

def creat_list_and_sort(path):
    """
    Create a list of files in the given path that end with '.npy' extension and sort them.

    Args:
        path (str): The path to the directory containing the files.

    Returns:
        list: A sorted list of file names.

    """
    file_list = [file for file in os.listdir(path) if file.endswith('.npy')] 
    sorted_list = sorted(file_list, key=extrair_valores)
    return sorted_list

def split_in_eps(array):
    """
    Splits the given array of indices into separate lists based on the episode number.

    Args:
        array (list): The array of indices.

    Returns:
        list: A list of lists, where each inner list contains indices belonging to the same episode.
    """
    eps = [[] for a in range(5)]
    for index in array:
        ep = int(index.split("_")[1]) - 1
        eps[ep].append(index)
    return eps

def equalize_length(eps):
    max_length = max(len(sublist) for sublist in eps)
    nullable_image = np.zeros((48, 64,3), dtype=np.uint8)
    for sublist in eps:
        sublist.extend([nullable_image] * (max_length - len(sublist)))
    return eps

def convert_path_in_scores(eps, path):
    eps_img = []
    for ep in eps:
        images = []
        for img in ep:
            images.append(np.load(os.path.join(path,img)))
        eps_img.append(images)
    return eps_img

def create_list_images(path):
    path_imgs = creat_list_and_sort(path)
    eps_path = split_in_eps(path_imgs)
    eps_img = convert_path_in_scores(eps_path, path)
    return eps_img

def calc_mean_per_scene(eps_scene, step):
    mean_per_scene = []
    for scene in eps_scene:
        mean_per_scene.append(scene[step])
    mean_per_scene = np.array(mean_per_scene)
    print(mean_per_scene.mean(axis=0))
    return mean_per_scene.mean(axis=0)

def calc_mean_all_scenes(scenes, episode, step):
    """
    Calculate the mean per step for all scenes.

    Args:
        scenes (list): A list of scenes.
        episode (int): The episode number.
        step (int): The step number.

    Returns:
        list: A list of mean values per step for each scene.
    """
    mean_per_step = []
    for scene in scenes:
        mean_per_step.append(calc_mean_per_scene(scene[episode], step))
    return mean_per_step

def calc_mean_all_scenes(scenes, episode, step):
    mean_per_setp = []
    for scene in scenes:
        mean_per_setp.append(calc_mean_per_scene(scene[episode], step))
    return mean_per_setp

def calc_mean_all_episodes_and_steps(scenes, shorter_lengths):
    """
    Calculate the mean per step for all episodes and steps.

    Args:
        scenes (list): List of scenes.
        shorter_lengths (list): List of shorter lengths for each episode.

    Returns:
        list: List of means per step for all episodes and steps.

    Examples:
        >>> scenes = [scene1, scene2, scene3]
        >>> shorter_lengths = [10, 15, 12]
        >>> calc_mean_all_episodes_and_steps(scenes, shorter_lengths)
        [[mean1, mean2, mean3, ...], [mean4, mean5, mean6, ...], [mean7, mean8, mean9, ...]]
    """
    means = []
    for episode, shorter in enumerate(shorter_lengths):
        means_per_step = []
        for step in range(shorter):
            means_per_step.append(calc_mean_all_scenes(scenes, episode, step))
        means.append(means_per_step)
    return means

def apply_color_map_in_list(datas):
    heatmap = []
    for data in datas:
        img = data * 255
        heatmap.append(cv2.applyColorMap(img.astype('uint8'), cv2.COLORMAP_JET))
    return heatmap

        
if __name__ == "__main__":
    model = "basic"

    basic_basic = create_list_images(f"grad-cam/scores/scores_basic/0_original/basic/basic")
    animated_animated = create_list_images(f"grad-cam/scores/scores_animated/0_original/animated/animated")
    caco_caco = create_list_images(f"grad-cam/scores/scores_caco/0_original/caco/caco")
    flat_flat= create_list_images(f"grad-cam/scores/scores_flat/0_original/flat/flat")

    basic_animated_ba_25 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_animated/ba_25")
    basic_animated_ba_50 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_animated/ba_50")
    basic_animated_ba_75 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_animated/ba_75")

    basic_caco_bc_25 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_caco/bc_25")
    basic_caco_bc_50 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_caco/bc_50")
    basic_caco_bc_75 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_caco/bc_75")

    basic_flat_bf_25 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_flat/bf_25")
    basic_flat_bf_50_1 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_flat/bf_50_1")
    basic_flat_bf_50_2 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_flat/bf_50_2")
    basic_flat_bf_75 = create_list_images(f"grad-cam/scores/scores_{model}/2_blend/basic_flat/bf_75")

    listas = [
    basic_basic, animated_animated, caco_caco, flat_flat,
    basic_animated_ba_25, basic_animated_ba_50, basic_animated_ba_75,
    basic_caco_bc_25, basic_caco_bc_50, basic_caco_bc_75,
    basic_flat_bf_25, basic_flat_bf_50_1, basic_flat_bf_50_2, basic_flat_bf_75
    ]
    shorter_lengths = [len(min(sublistas, key=len)) for sublistas in zip(*listas)]            
    teste = calc_mean_all_episodes_and_steps(listas, shorter_lengths)

    # nullable_image = np.zeros((48*3, 64*3,3), dtype=np.uint8)
    # width = 64*3
    # height = 48*3
    # for step in range(len(teste[0])):
    #     data = apply_color_map_in_list(teste[0][step])
    #     img_basic_basic = cv2.resize(data[0], (width, height))
    #     img_animated_animated = cv2.resize(data[1], (width, height))
    #     img_caco_caco = cv2.resize(data[2], (width, height))
    #     img_flat_flat = cv2.resize(data[3], (width, height))

    #     img_basic_animated_ba_25 = cv2.resize(data[4], (width, height))
    #     img_basic_animated_ba_50 = cv2.resize(data[5], (width, height))
    #     img_basic_animated_ba_75 = cv2.resize(data[6], (width, height))

    #     img_basic_caco_bc_25 = cv2.resize(data[7], (width, height))
    #     img_basic_caco_bc_50 = cv2.resize(data[8], (width, height))
    #     img_basic_caco_bc_75 = cv2.resize(data[9], (width, height))

    #     img_basic_flat_bf_25 = cv2.resize(data[10], (width, height))
    #     img_basic_flat_bf_50_1 = cv2.resize(data[11], (width, height))
    #     img_basic_flat_bf_50_2 = cv2.resize(data[12], (width, height))
    #     img_basic_flat_bf_75 = cv2.resize(data[13], (width, height))
        
    #     print(img_basic_basic.shape, img_basic_animated_ba_25.shape, img_basic_animated_ba_50.shape, img_basic_animated_ba_75.shape, img_basic_caco_bc_25.shape, img_basic_caco_bc_50.shape, img_basic_caco_bc_75.shape, img_basic_flat_bf_25.shape, img_basic_flat_bf_50_1.shape, img_basic_flat_bf_50_2.shape, img_basic_flat_bf_75.shape)

    #     # Criar composições
    #     composition1 = cv2.hconcat((img_basic_basic, img_basic_animated_ba_25, img_basic_animated_ba_50, img_basic_animated_ba_75,img_animated_animated,nullable_image))
    #     composition2 = cv2.hconcat((img_basic_basic, img_basic_caco_bc_25, img_basic_caco_bc_50, img_basic_caco_bc_75,img_caco_caco,nullable_image))
    #     composition3 = cv2.hconcat((img_basic_basic, img_basic_flat_bf_25, img_basic_flat_bf_50_1, img_basic_flat_bf_50_2, img_basic_flat_bf_75,img_flat_flat))

    #     # Compor as composições verticalmente
    #     final_composition = cv2.vconcat((composition1, composition2, composition3))

    #     # Exibir ou salvar a composição final
    #     cv2.imshow("Final Composition", final_composition)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()