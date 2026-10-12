from datasets.Sense_dataset import build_dataset as build_SENSE_dataset_train
from datasets.Sense_dataset import build_video_dataset as build_SENSE_dataset_test
from datasets.Metro_dataset import build_dataset as build_Metro_dataset_train
from datasets.Metro_dataset import build_video_dataset as build_Metro_dataset_test

def build_dataset(dataset_file, root, annotation_dir='', max_len=3000, train=False, step=15, interval=1, force_last=False):
    if train:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_train(root, annotation_dir, max_len, train=train, step=step) # step = 15
        elif dataset_file == 'Metro':
            return build_Metro_dataset_train(root, annotation_dir, max_len, train=train, step=step) # step = 1
    else:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_test(root, annotation_dir)
        elif dataset_file == 'Metro':
            return build_Metro_dataset_test(root, annotation_dir)