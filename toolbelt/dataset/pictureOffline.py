import cv2
import tqdm
import os
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
            img_height, img_width = img.shape[0], img.shape[1]
            new_height, new_width = mount_for_each_raw[0]*cut_into[0], mount_for_each_raw[1]*cut_into[1]
            img = cv2.resize(img, dsize=(new_width, new_height), fx=1, fy=1)
            for y in range(mount_for_each_raw[0]):
                for x in range(mount_for_each_raw[1]):
                    cut_img = img[y*cut_into[0]:(y+1)*cut_into[0], x*cut_into[1]:(x+1)*cut_into[1], :]
                    new_file_path = os.path.abspath(save_to) + f'\\{sample_type}'
                    if not os.path.exists(new_file_path):os.makedirs(new_file_path)
                    cv2.imwrite(f'{new_file_path}\\{count}_{y}_{x}.jpg', cut_img)
            count += 1
    return ans

def cutting_from_raw_picture(
  relative_path:str,
  save_to:str,
  cut_into=[300, 300], #[height, width]
  mount_for_each_raw = [3, 6],
):
    if os.name == 'nt': # if running on the windows.
        return __cutting_from_raw_picture_win(relative_path, save_to, cut_into, mount_for_each_raw)
    file_path = get_relative_paths(relative_path)
    ans = {}
    count = 0
    print('cutting_from_raw_picture running...')
    for each_path in tqdm.tqdm(file_path):
        sample_type = each_path.split('/')[0]
        full_path = relative_path+each_path
        if full_path[-3:] in ['jpg', 'png']:
            img = cv2.imread(relative_path+'/'+each_path)
            img_width = img.shape[1]
            img_heigth = img.shape[0]
            new_width, new_height = mount_for_each_raw[1]*cut_into[1], mount_for_each_raw[0]*cut_into[0]
            img = cv2.resize(img, dsize=(new_width, new_height), fx=1, fy=1)
            for y in range(mount_for_each_raw[0]):
                for x in range(mount_for_each_raw[1]):
                    cut_img = img[y*cut_into[0]:(y+1)*cut_into[0], x*cut_into[1]:(x+1)*cut_into[1], :]
                    new_file_path = os.path.abspath(save_to)+f'/{sample_type}/'
                    if not os.path.exists(new_file_path):os.makedirs(new_file_path)
                    cv2.imwrite(f'{new_file_path}{count}_{y}_{x}.jpg', cut_img)
            count += 1
    return ans