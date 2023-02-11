import os
import PIL.Image as Image
import torchvision.transforms

root_path = 'The root path of dataset'
drone_path = root_path + '/University-Release/test/gallery_drone_all'
target_path = root_path + '/University-Release/test/gallery_drone'

for dir_name in os.listdir(drone_path):
    folder_name = drone_path + '/' + dir_name + '/'
    target_folder_name = target_path + '/' + dir_name + '/'
    if not os.path.isdir(folder_name):
        continue
    if not os.path.isdir(target_folder_name):
        os.mkdir(target_folder_name)
        for file_name in os.listdir(folder_name):
            number = file_name.split('-')
            number = number[1]
            number = int(number[0:2])
            # We only use the high-altitude drone-view images as the reference.
            if number >= 37:
                os.system('cp %s %s' % (folder_name + file_name, target_folder_name + file_name))


