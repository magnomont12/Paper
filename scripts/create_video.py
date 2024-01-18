import os
import cv2
import argparse

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "path_with_images",
    help="CFG file with settings of the ViZDoom scenario.")

def separar_listas(lista):
    max_episodes = max(lista, key=lambda item: int(item.split("_")[1])).split("_")[1]
    listas = [[] for _ in range(int(max_episodes))]
    print(max_episodes)
    for item in lista:
        partes = item.split("_")
        numero = int(partes[1])
        listas[numero - 1].append(item)
    return listas

def criar_video(images):
    images = sorted(images, key=lambda item: int(item.split("_")[2]))
    episode = images[0].split("_")[1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    height = height*8
    width = width*8
    video = cv2.VideoWriter(os.path.join(image_folder,f'video_{episode}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)
    video.release()
    cv2.destroyAllWindows()


image_folder = parser.parse_args().path_with_images
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
episodes_images = separar_listas(images)
for episode in episodes_images:
    criar_video(episode)