'''
tool collect data
run script outside of data folder
python3 collect.py <data_path>
'''
import sys
import os
import shutil

img_list = ['side_1_1_square_1024.jpg',
            'side_1_2_square_1024.jpg',
            'side_2_1_square_1024.jpg',
            'side_2_2_square_1024.jpg',
            ]

device_list  = ['0','1','2','3','4','5','6']

data_path = sys.argv[1]
its = os.listdir(data_path)
ids = []
# check id folder
for it in its:
    id_path = os.path.join(data_path,it)
    if os.path.isdir(id_path):
        ids.append(it)

# create destination folder
if not os.path.exists('dst'):
    os.mkdir('dst')
else:
    print('please remove dst first!')
    exit(-1)
    
# list all ids
for id in ids:
    id_path = os.path.join(data_path,id)
    print('checking: ',id_path)
    
    things  = os.listdir(id_path)
    # find folder device in things
    devices = []
    for thing in things:
        if os.path.isdir(os.path.join(id_path,thing)) and thing in device_list:
            devices.append(thing)
    
    # copy img in each device
    for device in devices:
        device_path = os.path.join(id_path,device)
        # copy img in img_list:
        for img in img_list:
            try:
                img_name = f'image_detect_device_{device}_'+img
                img_name_dst = f'{id}_{device}_{img}'
                # copy img
                shutil.copy(os.path.join(device_path,img_name),os.path.join('dst',img_name_dst))
            except Exception as e:
                print(e)
                print('error:',img_name)
            
print('done!')
    
    

