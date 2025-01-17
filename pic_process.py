import numpy as np
import cv2
import os
import tqdm
import torchvision as tcv
from PIL import Image
# This is a python module
"""
This Module Will Be DEPRECIATED!!!
Please using toolbelt.dataset.pictureOffline
"""

def get_relative_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, start=directory)
            file_paths.append(relative_path)
    return file_paths


# DON'T USE THIS FUNCTION DIRECTLY!!
def __cutting_from_raw_picture_win(
    relative_path:str,
    save_to:str,
    cut_into = [300, 300], #[height, width]
    mount_for_each_raw = [3, 6],
    process:'function' = None, # process is a function that will be used to process the image, and input must be a cv2 image.
    keep_original_name = False, # you will keep the original file name with new file name, your file name will be incredibly long. OMG, I don't know why you want to do this.
):
    relative_path = os.path.abspath(relative_path)
    file_path = get_relative_paths(relative_path)
    ans = {}
    count = 0
    print('cutting_from_raw_picture running...')
    for each_path in tqdm.tqdm(file_path):
        sample_type = each_path.split('\\')[0]
        full_path = relative_path+'\\'+each_path
        abspath = os.path.abspath(full_path)
        if abspath.split('.')[-1] in ['jpg', 'png']:
            # Process for each 
            img = cv2.imread(abspath)
            originalName = each_path.split('\\')[-1]
            new_height, new_width = mount_for_each_raw[0]*cut_into[0], mount_for_each_raw[1]*cut_into[1]
            img = cv2.resize(img, dsize=(new_width, new_height), fx=1, fy=1)
            for y in range(mount_for_each_raw[0]):
                for x in range(mount_for_each_raw[1]):
                    cut_img = img[y*cut_into[0]:(y+1)*cut_into[0], x*cut_into[1]:(x+1)*cut_into[1], :]
                    if process!=None:
                        cut_img = process(cut_img)
                    new_file_path = os.path.abspath(save_to) + f'\\{sample_type}'
                    if not os.path.exists(new_file_path):os.makedirs(new_file_path)
                    cv2.imwrite(f'{new_file_path}\\{f"{originalName}_" if keep_original_name else ""}{count}_{y}_{x}.jpg', cut_img)
            count += 1
    return ans

def cutting_from_raw_picture(
  relative_path:str,
  save_to:str,
  cut_into=[300, 300], #[height, width]
  mount_for_each_raw = [3, 6],
  process:'function' = None, # process is a function that will be used to process the image, and input must be a cv2 image.
  keep_original_name = False,
):
    if os.name == 'nt': # if running on the windows.
        return __cutting_from_raw_picture_win(relative_path, save_to, cut_into, mount_for_each_raw, process, keep_original_name)
    file_path = get_relative_paths(relative_path)
    ans = {}
    count = 0
    print('cutting_from_raw_picture running...')
    for each_path in tqdm.tqdm(file_path):
        sample_type = each_path.split('/')[0]
        full_path = relative_path+each_path
        if full_path[-3:] in ['jpg', 'png']:
            img = cv2.imread(relative_path+'/'+each_path)
            originalName = each_path.split('/')[-1]
            new_width, new_height = mount_for_each_raw[1]*cut_into[1], mount_for_each_raw[0]*cut_into[0]
            img = cv2.resize(img, dsize=(new_width, new_height), fx=1, fy=1)
            for y in range(mount_for_each_raw[0]):
                for x in range(mount_for_each_raw[1]):
                    cut_img = img[y*cut_into[0]:(y+1)*cut_into[0], x*cut_into[1]:(x+1)*cut_into[1], :]
                    if process!=None:
                        cut_img = process(cut_img)
                    new_file_path = os.path.abspath(save_to)+f'/{sample_type}/'
                    if not os.path.exists(new_file_path):os.makedirs(new_file_path)
                    cv2.imwrite(f'{new_file_path}{f"{originalName}_" if keep_original_name else ""}{count}_{y}_{x}.jpg', cut_img)
            count += 1
    return ans


# This class is used for torchvision.transforms.

def __process(img:np.ndarray):
    ans = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ans = cv2.medianBlur(ans, 5)
    ans = cv2.adaptiveThreshold(ans, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 2)
    ans = cv2.medianBlur(ans, 5)
    return ans

if __name__ == '__main__':
    # 第一个参数是原来数据集的路径，第二个参数是切完后数据集的路径
    cutting_from_raw_picture(
        #'C:\\Users\\XY\\Desktop\\python_cut\\picture',
        #'C:\\Users\\XY\\Desktop\\python_cut\\result',
        cut_into=[300, 300],
        mount_for_each_raw=[3,6],   # 如果不需要切图的话，请将这里的[3,6]改为[1,1]就可以了
        process = __process,
        keep_original_name=False,
    )