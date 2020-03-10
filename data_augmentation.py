from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os

image_path = ##ImagePath

datagen = ImageDataGenerator(rescale=1./255,
                            rotation_range=0.5,
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='nearest')

for i in os.listdir(image_path):
    filename_in_dir = []
    image_folder_list=i
    for r,d,f in os.walk(image_path+image_folder_list):
        for fname in f:
            full_fname = os.path.join(r, fname)
            filename_in_dir.append(full_fname)

    for file_image in filename_in_dir:
        print(file_image)
        img = image.load_img(file_image)
        x = image.img_to_array(img)
        x = x.reshape((1,)+x.shape)

        num = 0

        for batch in datagen.flow(x,save_to_dir=image_path+image_folder_list, save_prefix='augmented', save_format='png'):
            num += 1
            if num > 1: # total 4 image augmentation
                break
