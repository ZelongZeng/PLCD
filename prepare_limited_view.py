import os
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--root_path', default='.', type=str, help='Input Your Root Path Of The Dataset')
opt = parser.parse_args()

drone_path = opt.root_path + '/University/University-Release/test/gallery_drone'
target_low_path = opt.root_path+ '/University/University-Release/test/gallery_drone_low'
target_middle_path = opt.root_path+ '/University/University-Release/test/gallery_drone_middle'
target_high_path = opt.root_path+ '/University/University-Release/test/gallery_drone_high'

if not os.path.isdir(target_low_path):
    os.mkdir(target_low_path)
if not os.path.isdir(target_middle_path):
    os.mkdir(target_middle_path)
if not os.path.isdir(target_high_path):
    os.mkdir(target_high_path)

for dir_name in os.listdir(drone_path):
    folder_name = drone_path + '/' + dir_name + '/'
    target_low_folder_name = target_low_path + '/' + dir_name + '/'
    target_middle_folder_name = target_middle_path + '/' + dir_name + '/'
    target_high_folder_name = target_low_path + '/' + dir_name + '/'
    if not os.path.isdir(folder_name):
        continue
    if not os.path.isdir(target_low_folder_name):
        os.mkdir(target_low_folder_name)
    if not os.path.isdir(target_middle_folder_name):
        os.mkdir(target_middle_folder_name)
    if not os.path.isdir(target_high_folder_name):
        os.mkdir(target_high_folder_name)

    for file_name in os.listdir(folder_name):
        number = file_name.split('-')
        number = number[1]
        number = int(number[0:2])
        if number >= 37:
            os.system('cp %s %s' % (folder_name + file_name, target_low_folder_name + file_name))
        if number > 18 and number < 37:
            os.system('cp %s %s' % (folder_name + file_name, target_middle_folder_name + file_name))
        if number <= 18:
            os.system('cp %s %s' % (folder_name + file_name, target_high_folder_name + file_name))



