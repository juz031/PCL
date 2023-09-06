import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(description='Data sampling')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--shapes', type=str, metavar='SHAPEDIR',
                    help='path to shapes')                    
parser.add_argument('--percent', default=75, type=int, metavar='P',
                    help='the percent of whole dataset')

def main():
    args = parser.parse_args()

    percent = args.percent / 100

    train_dir = os.path.join(args.data, 'train')
    shapes_dir = os.path.join(args.shapes, 'train')

    new_img_dir = '/user_data/junruz/imagenette_subset'
    if not os.path.exists(new_img_dir):
        os.mkdir(new_img_dir)

    new_shape_dir = '/user_data/junruz/imagenette_subset_masks'
    if not os.path.exists(new_shape_dir):
        os.mkdir(new_shape_dir)

    for n in range(5):
        img_save_dir = os.path.join(new_img_dir, str(args.percent)+ '_{}'.format(n))
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)
        
        shape_save_dir = os.path.join(new_shape_dir, str(args.percent)+ '_{}'.format(n))
        if not os.path.exists(shape_save_dir):
            os.mkdir(shape_save_dir)

        new_train_dir = os.path.join(img_save_dir, 'train')
        if not os.path.exists(new_train_dir):
            os.mkdir(new_train_dir)
        
        shape_train_dir = os.path.join(shape_save_dir, 'train')
        if not os.path.exists(shape_train_dir):
            os.mkdir(shape_train_dir)
        

        category_list = os.listdir(train_dir)
        for category in category_list:
            print(category)
            category_dir = os.path.join(new_train_dir, category)
            if not os.path.exists(category_dir):
                os.mkdir(category_dir)
            
            shape_category_dir = os.path.join(shape_train_dir, category)
            if not os.path.exists(shape_category_dir):
                os.mkdir(shape_category_dir)
            
            img_dir = os.path.join(train_dir, category)
            shape_dir = os.path.join(shapes_dir, category)
            img_list = os.listdir(img_dir)
            subset_list = random.sample(img_list, int(percent * len(img_list)))
            print('Number of imgs: {}'.format(len(subset_list)))
            for img in subset_list:
                shape = os.path.splitext(img)[0] + '.png'
                shutil.copy2(os.path.join(img_dir, img), os.path.join(category_dir, img))
                shutil.copy2(os.path.join(shape_dir, shape), os.path.join(shape_category_dir, shape))



if __name__ == '__main__':
    main()