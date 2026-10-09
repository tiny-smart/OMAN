import numpy as np
import json
import os

with open("outputs/json/video_results_test.json", "r") as f:
    video_results = json.load(f)

anno_root = "/data/bjwang/WuhanMetro/json"

D0 = {'gt_video_num_list':[], 'pred_video_num_list':[], 'gt_video_len_list':[]}
D1 = {'gt_video_num_list':[], 'pred_video_num_list':[], 'gt_video_len_list':[]}

scene = {
    '站台':['val-02','val-03','val-09','val-10','val-11','val-12','val-14','test-01','test-02','test-12','test-13','test-15','test-16','test-18','test-20'],
    '换乘通道':['val-01','val-05','val-07','val-15','test-03','test-07','test-08',],
    '闸机':['val-06','val-08','val-13','test-17','test-09','test-14','test-19'],
    '大厅':['val-04','test-11',],
    '扶梯':['test-04','test-06',],
    '出入口':['test-05','test-10',]
}
scene_results = {
    'gt_video_num_list':{'站台':[],'换乘通道':[],'闸机':[],'大厅':[],'扶梯':[],'出入口':[]},
    'pred_video_num_list':{'站台':[],'换乘通道':[],'闸机':[],'大厅':[],'扶梯':[],'出入口':[]},
    'gt_video_len_list':{'站台':[],'换乘通道':[],'闸机':[],'大厅':[],'扶梯':[],'出入口':[]}
}
scene_WRAE = {'站台':float,'换乘通道':float,'闸机':float,'大厅':float,'扶梯':float,'出入口':float}




rmae_list = []
gt_video_num_list = []
gt_video_len_list = []
pred_video_num_list = []
pred_matched_num_list = []
gt_matched_num_list = []
WCA_total_up = 0
WCA_total_down = 0
r2_up = 0
r2_down = []
total_frame = 0
loc_mae = []

# remove_list = ['test-13', 'test-05', 'val-03', 'val-01', 'val-09']
remove_list = []
for video_name in video_results:
    if video_name not in remove_list:
        pedestrian_num = 0
        inflow_num = 0
        outflow_num = 0
        video_len = 0
        total_num = 0
        gt_cnt = []
        pred_cnt = []


        annotation_dir = os.path.join(anno_root, video_name)
        annotations_paths = os.listdir(annotation_dir)
        annotations_paths = [os.path.join(annotation_dir, annotation) for annotation in annotations_paths]


        """ GROUND TRUTH """
        for annotation_path in annotations_paths:
            file_index = []
            file_list = os.listdir(annotation_dir)
            for i in range(len(file_list)):
                # 把文件名里的数字提出来排序
                split_file_list = file_list[i].rsplit('_', 1)
                if len(split_file_list) == 1:
                    prefix = ''
                else:
                    prefix = split_file_list[0]

                idx = file_list[i].split('.')[0].split("_")[-1]
                file_index.append(int(idx))
                idx_len = len(idx)
                file_index = sorted(file_index)
                total_label = []  # 统计所有文件的label标签数量
                cnt = 0

        for i in range(len(file_index)):
            file = file_index[i]
            mask_num = 0
            anno_path = os.path.join(annotation_dir, prefix +'_'+ str(file).zfill(idx_len) + '.json')
            if i == 0:
                with open(anno_path, "r") as f:
                    label_list = json.load(f)
                    cnt = len(label_list['shapes'])
                    for j in range(cnt):
                        if label_list['shapes'][j]['label'] in ['mask']:
                            mask_num += 1
                            continue
                inflow_num += cnt
                total_num += cnt
            else:
                with open(anno_path, "r") as f:
                    label_list = json.load(f)
                    cnt = len(label_list['shapes'])
                    for j in range(cnt):
                        if label_list['shapes'][j]['label'] in ['pedestrian', 'Pedestrian', 'pedestrain']:
                            pedestrian_num += 1
                        elif label_list['shapes'][j]['label'] == 'inflow' or label_list['shapes'][j][
                            'label'] == 'Inflow':
                            inflow_num += 1
                        elif label_list['shapes'][j]['label'] == 'outflow' or label_list['shapes'][j][
                            'label'] == 'Outflow':
                            outflow_num += 1
                        elif label_list['shapes'][j]['label'] == 'both' or label_list['shapes'][j][
                            'label'] == 'Both':
                            inflow_num += 1
                            outflow_num += 1
                        elif label_list['shapes'][j]['label'] == 'mask':
                            mask_num += 1
                            continue
                        else:
                            # print(label_list['shapes'][j]['label'])
                            continue
            total_num += cnt
            total_num -= mask_num
            gt_cnt.append(cnt - mask_num)


        """ PREDICTION """
        info = video_results[video_name]
        gt_video_num = inflow_num
        pred_video_num = info["video_num"]
        pred_video_num_list.append(pred_video_num)
        gt_video_num_list.append(gt_video_num)
        gt_video_len_list.append(info["frame_num"])
        rmae_list.append(abs((pred_video_num-gt_video_num)/gt_video_num))
        for i in range(len(info["pos_lists"])):
            pred_cnt.append(info["pos_lists"][i])

        # assert len(pred_cnt) == len(gt_cnt)
        wca_up = 0
        for i in range(len(pred_cnt)):
            wca_up += abs(len(pred_cnt[i]) - gt_cnt[i])
            r2_up += pow(len(pred_cnt[i]) - gt_cnt[i], 2)
            r2_down.append(gt_cnt[i])
            loc_mae.append(abs(len(pred_cnt[i]) - gt_cnt[i]))
        WCA_total_up += wca_up
        wca_down = sum(gt_cnt)
        WCA_total_down += wca_down
        WCA = 1 - wca_up / wca_down

        dens = total_num/info["frame_num"]
        total_frame += info["frame_num"]
        print(f"{video_name}, pred_num:{pred_video_num}, gt_num:{gt_video_num}, RMAE:{(pred_video_num-gt_video_num)*100/gt_video_num:.2f}%, "
              f"WCA:{WCA:.2f}, Density:{dens:.2f}, frame:{info['frame_num']}")

        if dens>50:
            D1['gt_video_num_list'].append(gt_video_num)
            D1['pred_video_num_list'].append(pred_video_num)
            D1['gt_video_len_list'].append(info["frame_num"])
        elif dens<=50:
            D0['gt_video_num_list'].append(gt_video_num)
            D0['pred_video_num_list'].append(pred_video_num)
            D0['gt_video_len_list'].append(info["frame_num"])

        for scene_name in ['站台','换乘通道','闸机','大厅','扶梯','出入口']:
            if video_name in scene[scene_name]:
                scene_results['gt_video_num_list'][scene_name].append(gt_video_num)
                scene_results['pred_video_num_list'][scene_name].append(pred_video_num)
                scene_results['gt_video_len_list'][scene_name].append(info["frame_num"])
                break

MAE = np.mean(np.abs(np.array(gt_video_num_list) - np.array(pred_video_num_list)))
MSE = np.mean(np.square(np.array(gt_video_num_list) - np.array(pred_video_num_list)))
WRAE = np.sum(
    np.abs(np.array(gt_video_num_list) - np.array(pred_video_num_list)) * np.array(gt_video_len_list) / np.array(
        gt_video_num_list) / np.sum(gt_video_len_list))
RMSE = np.sqrt(MSE)
RMAE = np.array(rmae_list).mean()

R2_total_down = 0
R2_total_up = r2_up
r2_down_mean = np.array(r2_down).mean()
for i in range(len(r2_down)):
    R2_total_down += pow((r2_down[i] - r2_down_mean), 2)
R2 = 1 - R2_total_up/R2_total_down
WCA_total = 1 - WCA_total_up/WCA_total_down

print(f"MAE:{MAE:.2f}, MSE:{MSE:.2f}, RMSE:{RMSE:.2f}, RMAE:{RMAE * 100:.2f}%, WRAE:{WRAE * 100:.2f}%, LOC_MAE:{sum(loc_mae)/len(loc_mae):.2f}, WCA:{WCA_total * 100:.2f}%, R2:{R2 * 100:.2f}%")

D0_WRAE = np.sum(
    np.abs(np.array(D0['gt_video_num_list']) - np.array(D0['pred_video_num_list'])) * np.array(D0['gt_video_len_list']) / np.array(
        D0['gt_video_num_list']) / np.sum(D0['gt_video_len_list']))
D1_WRAE = np.sum(
    np.abs(np.array(D1['gt_video_num_list']) - np.array(D1['pred_video_num_list'])) * np.array(D1['gt_video_len_list']) / np.array(
        D1['gt_video_num_list']) / np.sum(D1['gt_video_len_list']))
print(f"D0 (sparse):{D0_WRAE*100:.2f}%, "
      f"D1 (dense):{D1_WRAE*100:.2f}%, ")

for scene_name in ['站台','换乘通道','闸机','大厅','扶梯','出入口']:
    scene_WRAE[scene_name] = \
        f"{np.sum(np.abs(np.array(scene_results['gt_video_num_list'][scene_name]) - np.array(scene_results['pred_video_num_list'][scene_name])) * np.array(scene_results['gt_video_len_list'][scene_name]) / np.array(scene_results['gt_video_num_list'][scene_name]) / np.sum(scene_results['gt_video_len_list'][scene_name])) * 100:.2f}%"
print(scene_WRAE)


# FPS
# MDC:2.46; OMAN++:7.54; OMAN:6.82; CGNet:14.45;

# Param (M)
# OMAN++: 81.20; MDC: 31.5;

# GFLOPs
# OMAN++: ; MDC: 2325.59; FMDC: 1109.77

# 1 3: MAE:133.10, MSE:79308.60, RMSE:281.62, RMAE:28.57%, WRAE:28.67%, LOC_MAE:10.60, WCA:60.72%, R2:65.28%
# 2 3: MAE:119.45, MSE:54951.05, RMSE:234.42, RMAE:28.50%, WRAE:27.59%, LOC_MAE:10.67, WCA:60.47%, R2:62.55%
# 1 2: MAE:57.80, MSE:5535.10, RMSE:74.40, RMAE:25.35%, WRAE:27.63%, LOC_MAE:8.11, WCA:69.93%, R2:77.71%