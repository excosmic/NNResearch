import platform
import os
def path2name(path:str) -> str:
    match platform.system():
        case 'Windows':
            return path.split('\\')[-1]
        case 'Linux':
            return path.split('/')[-1]
        case 'Darwin':
            return path.split('/')[-1]
    raise ValueError(f'Unsupported platform: {path}')

def get_relative_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, start=directory)
            file_paths.append(relative_path)
    return file_paths

def get_absolute_paths(directory):
    directory = os.path.abspath(directory)
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths += [file_path]
    return file_paths
