import numpy as np
import json
import os

with open("outputs/json/video_results_test.json", "r") as f:
    video_results = json.load(f)

anno_root = "/data/SENSE/label_list_all"

with open('/data/SENSE/scene_label.txt', 'r') as f:
    lines = f.readlines()
scene_label = {}
for line in lines:
    line = line.rstrip().split(' ')
    scene_label.update({line[0]: [int(i) for i in line[1:]]})
scene_dict = {'D0':[], 'D1':[], 'D2':[], 'D3':[], 'D4':[]}


rmae_list = []
gt_video_num_list = []
gt_video_len_list = []
pred_video_num_list = []
pred_matched_num_list = []
gt_matched_num_list = []
for video_name in video_results:
    video_len = 0
    anno_path = os.path.join(anno_root, video_name + ".txt")
    with open(anno_path, "r") as f:
        lines = f.readlines()
        all_ids = set()

        for line in lines:
            line = line.strip().split(" ")
            data = [float(x) for x in line[3:] if x != ""]
            if len(data) > 0:
                data = np.array(data)
                data = np.reshape(data, (-1, 7))
                ids = data[:, 6].reshape(-1, 1)
                for id in ids:
                    all_ids.add(int(id[0]))
    info = video_results[video_name]
    gt_video_num = len(all_ids)
    pred_video_num = info["video_num"]
    pred_video_num_list.append(pred_video_num)
    gt_video_num_list.append(gt_video_num)
    gt_video_len_list.append(info["frame_num"])
    rmae_list.append(abs((pred_video_num-gt_video_num)/gt_video_num))
    # if abs(pred_video_num-gt_video_num)/gt_video_num>0.126:
    print(video_name, "pred_video_num:",pred_video_num,"gt_video_num:", gt_video_num, "RMAE:",(pred_video_num-gt_video_num)/gt_video_num)

    scene_l = scene_label[video_name]
    if scene_l[3] == 0: scene_dict['D0'].append(abs(gt_video_num-pred_video_num))
    if scene_l[3] == 1: scene_dict['D1'].append(abs(gt_video_num-pred_video_num))
    if scene_l[3] == 2: scene_dict['D2'].append(abs(gt_video_num-pred_video_num))
    if scene_l[3] == 3: scene_dict['D3'].append(abs(gt_video_num-pred_video_num))
    if scene_l[3] == 4: scene_dict['D4'].append(abs(gt_video_num-pred_video_num))

for i in range(len(pred_video_num_list)):
    pred_video_num_list[i] = pred_video_num_list[i]
MAE = np.mean(np.abs(np.array(gt_video_num_list) - np.array(pred_video_num_list)))
MSE = np.mean(np.square(np.array(gt_video_num_list) - np.array(pred_video_num_list)))
WRAE = np.sum(
    np.abs(np.array(gt_video_num_list) - np.array(pred_video_num_list)) * np.array(gt_video_len_list) / np.array(
        gt_video_num_list) / np.sum(gt_video_len_list))
RMSE = np.sqrt(MSE)
RMAE = np.array(rmae_list).mean()
print(f"MAE:{MAE:.2f}, MSE:{MSE:.2f}, WRAE:{WRAE * 100:.2f}%, RMSE:{RMSE:.2f}, RMAE:{RMAE * 100:.2f}%")
print(f"D0:{sum(scene_dict['D0'])/len(scene_dict['D0']):.2f}, "
      f"D1:{sum(scene_dict['D1'])/len(scene_dict['D1']):.2f}, "
      f"D2:{sum(scene_dict['D2'])/len(scene_dict['D2']):.2f}, "
      f"D3:{sum(scene_dict['D3'])/len(scene_dict['D3']):.2f}, "
      f"D4:{sum(scene_dict['D4'])/len(scene_dict['D4']):.2f}, ")