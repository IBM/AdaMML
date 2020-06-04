
"""
For each dataset, the following fields are required:
  - num_classes: number of classes
  - train_list_name: the filename of train list
  - val_list_name: the filename of val list
  - filename_separator: the separator used in train/val/test list
  - image_tmpl: the template of images in the video folder
  - filter_video: the threshold to remove videos whose frame number is less than this value

Those are optional:
  - test_list_name: the filename of test list
  - label_file: name of classes, used to map the prediction from a model to real label name

"""


DATASET_CONFIG = {
    'activitynet': {
        'num_classes': 200,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': 'image_{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    },
    'fcvid': {
        'num_classes': 239,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': 'image_{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    },
    'kinetics-sounds': {
        'num_classes': 31,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    },
    'mini-sports1m': {
        'num_classes': 487,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    }    
}


def get_dataset_config(dataset):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name']
    val_list_name = ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    if test_list_name is not None:
        test_list_name = test_list_name
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file


if __name__ == "__main__":
    print(get_dataset_config('activitynet'))
