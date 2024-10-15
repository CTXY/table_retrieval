import os
path_parts = os.path.realpath(__file__).split(os.sep)

# 通过切片移除最后两个目录，从而获取到根目录的路径
root_path = os.sep.join(path_parts[:-4])

print("Project Root Path:", root_path)