from datasets.Sense_dataset import build_dataset as build_SENSE_dataset_train
from datasets.HT21_dataset import build_dataset as build_HT21_dataset_train
from datasets.Sense_dataset import build_video_dataset as build_SENSE_dataset_test
from datasets.HT21_dataset import build_video_dataset as build_HT21_dataset_test
from datasets.Metro_dataset import build_dataset as build_Metro_dataset_train
from datasets.Metro_dataset import build_video_dataset as build_Metro_dataset_test
from datasets.UAVVIC_dataset import build_dataset as build_UAVVIC_dataset_train
from datasets.UAVVIC_dataset import build_video_dataset as build_UAVVIC_dataset_test
from datasets.MovingDroneCrowd_dataset import build_dataset as build_MovingDroneCrowd_dataset_train
from datasets.MovingDroneCrowd_dataset import build_video_dataset as build_MovingDroneCrowd_dataset_test

def build_dataset(dataset_file, root, annotation_dir='', max_len=3000, train=False, step=15, interval=1, force_last=False):
    if train:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_train(root, annotation_dir, max_len, train=train, step=step) # step = 15
        elif dataset_file == 'HT21':
            return build_HT21_dataset_train(root, max_len, train=train, step=step) # step = 75
        elif dataset_file == 'Metro':
            return build_Metro_dataset_train(root, annotation_dir, max_len, train=train, step=step) # step = 1
        elif dataset_file == 'UAVVIC':
            return build_UAVVIC_dataset_train(annotation_dir, root, max_len, train=train, step=step)
        elif dataset_file == 'Drone':
            return build_MovingDroneCrowd_dataset_train(root, annotation_dir, max_len, train=train, step=step)
    else:
        if dataset_file == 'SENSE':
            return build_SENSE_dataset_test(root, annotation_dir)
        elif dataset_file == 'HT21':
            return build_HT21_dataset_test(root)
        elif dataset_file == 'Metro':
            return build_Metro_dataset_test(root, annotation_dir)
        elif dataset_file == 'UAVVIC':
            return build_UAVVIC_dataset_test(annotation_dir, root)
        elif dataset_file == 'Drone':
            return build_MovingDroneCrowd_dataset_test(root, annotation_dir)