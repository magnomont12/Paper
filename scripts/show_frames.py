import os
import cv2
import numpy as np

def extrair_valores(string):
    splits = string.split("_")
    x = int(splits[1])
    y = int(splits[2])
    return x, y

def creat_list_and_sort(path):
    file_list = [file for file in os.listdir(path) if file.endswith('.png')] 
    sorted_list = sorted(file_list, key=extrair_valores)
    return sorted_list

def split_in_eps(array):
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

def convert_path_in_images(eps, path):
    eps_img = []
    for ep in eps:
        images = []
        for img in ep:
            images.append(cv2.imread(os.path.join(path,img)))
        eps_img.append(images)
    return eps_img

def create_list_images(path):
    path_imgs = creat_list_and_sort(path)
    eps_path = split_in_eps(path_imgs)
    eps_img = convert_path_in_images(eps_path, path)
    return eps_img

def resize_images(*images):
    max_height = max(img.shape[0]*3 for img in images)
    max_width = max(img.shape[1]*3 for img in images)
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]
    return resized_images

        


if __name__ == "__main__":
    model = "basic"
    # caco
    basic_basic = create_list_images(f"grad-cam/blends/blend_basic/0_original/basic/basic")
    animated_animated = create_list_images(f"grad-cam/blends/blend_animated/0_original/animated/animated")
    caco_caco = create_list_images(f"grad-cam/blends/blend_caco/0_original/caco/caco")
    flat_flat= create_list_images(f"grad-cam/blends/blend_flat/0_original/flat/flat")

    basic_animated_ba_25 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_animated/ba_25")
    basic_animated_ba_50 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_animated/ba_50")
    basic_animated_ba_75 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_animated/ba_75")

    basic_caco_bc_25 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_caco/bc_25")
    basic_caco_bc_50 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_caco/bc_50")
    basic_caco_bc_75 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_caco/bc_75")

    basic_flat_bf_25 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_flat/bf_25")
    basic_flat_bf_50_1 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_flat/bf_50_1")
    basic_flat_bf_50_2 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_flat/bf_50_2")
    basic_flat_bf_75 = create_list_images(f"grad-cam/blends/blend_{model}/2_blend/basic_flat/bf_75")


    ep = 1
    teste = equalize_length([basic_basic[ep], basic_animated_ba_25[ep], basic_animated_ba_50[ep], basic_animated_ba_75[ep],
                             basic_caco_bc_25[ep], basic_caco_bc_50[ep], basic_caco_bc_75[ep], basic_flat_bf_25[ep],
                             basic_flat_bf_50_1[ep], basic_flat_bf_50_2[ep], basic_flat_bf_75[ep], animated_animated[ep], caco_caco[ep], flat_flat[ep]])

    
    nullable_image = np.zeros((48*3, 64*3,3), dtype=np.uint8)
    width = 64*3
    height = 48*3
    for a in range(len(teste[0])):
        img_basic_basic = cv2.resize(teste[0][a], (width, height))

        img_basic_animated_ba_25 = cv2.resize(teste[1][a], (width, height))
        img_basic_animated_ba_50 = cv2.resize(teste[2][a], (width, height))
        img_basic_animated_ba_75 = cv2.resize(teste[3][a], (width, height))

        img_basic_caco_bc_25 = cv2.resize(teste[4][a], (width, height))
        img_basic_caco_bc_50 = cv2.resize(teste[5][a], (width, height))
        img_basic_caco_bc_75 = cv2.resize(teste[6][a], (width, height))

        img_basic_flat_bf_25 = cv2.resize(teste[7][a], (width, height))
        img_basic_flat_bf_50_1 = cv2.resize(teste[8][a], (width, height))
        img_basic_flat_bf_50_2 = cv2.resize(teste[9][a], (width, height))
        img_basic_flat_bf_75 = cv2.resize(teste[10][a], (width, height))

        img_animated_animated = cv2.resize(teste[11][a], (width, height))
        img_caco_caco = cv2.resize(teste[12][a], (width, height))
        img_flat_flat = cv2.resize(teste[13][a], (width, height))
        
        print(img_basic_basic.shape, img_basic_animated_ba_25.shape, img_basic_animated_ba_50.shape, img_basic_animated_ba_75.shape, img_basic_caco_bc_25.shape, img_basic_caco_bc_50.shape, img_basic_caco_bc_75.shape, img_basic_flat_bf_25.shape, img_basic_flat_bf_50_1.shape, img_basic_flat_bf_50_2.shape, img_basic_flat_bf_75.shape)

        # Criar composições
        composition1 = cv2.hconcat((img_basic_basic, img_basic_animated_ba_25, img_basic_animated_ba_50, img_basic_animated_ba_75,img_animated_animated,nullable_image))
        composition2 = cv2.hconcat((img_basic_basic, img_basic_caco_bc_25, img_basic_caco_bc_50, img_basic_caco_bc_75,img_caco_caco,nullable_image))
        composition3 = cv2.hconcat((img_basic_basic, img_basic_flat_bf_25, img_basic_flat_bf_50_1, img_basic_flat_bf_50_2, img_basic_flat_bf_75,img_flat_flat))

        # Compor as composições verticalmente
        final_composition = cv2.vconcat((composition1, composition2, composition3))

        # Exibir ou salvar a composição final
        cv2.imshow("Final Composition", final_composition)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # Carregando a primeira imagem de cada lista
    # primeiras_imagens = [lista for lista in teste]
    # print(len(primeiras_imagens[0]))

    # # Obtendo as dimensões das imagens
    # altura, largura, _ = primeiras_imagens[0].shape

    # # Criando uma imagem que será a composição das primeiras imagens
    # imagem_composta = np.zeros((altura, largura * len(primeiras_imagens), 3), dtype=np.uint8)

    # # Compondo a imagem final
    # for i, imagem in enumerate(primeiras_imagens):
    #     imagem_composta[:, i * largura:(i + 1) * largura, :] = imagem

    # # Exibindo a imagem composta
    # cv2.imshow("Primeiras Imagens", imagem_composta)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



