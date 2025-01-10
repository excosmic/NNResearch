import os
def get_relative_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, start=directory)
            file_paths.append(relative_path)
    return file_paths

