import os
import pandas as pd
import numpy as np
import time

target_list = ['4p1b_01A2', '5p2b_01A1', '5p4b_01A2', '5p5b_03A1', '7p3b_02M']
target_elapsed_time = []

for target in target_list:
    video_path = os.path.join('..', 'videos', target, target + '.m4v')
    gt_csv_path = os.path.join('..', 'videos', target, target + '.csv')

    if not (os.path.exists(video_path) and os.path.exists(gt_csv_path)):
        print('Missing video or ground truth csv for target {}'.format(target))
        continue

    begin = time.time()
    os.system('python solution %s %s' % (video_path, gt_csv_path))
    end = time.time()
    target_elapsed_time.append(end - begin)

for i in range(len(target_list)):
    target, elapsed_time = target_list[i], target_elapsed_time[i]
    gt_csv_path = os.path.join('..', 'videos', target, 'correct_%s.MP4.csv' % target)
    output_csv_path = os.path.join('outputs', target + '_out.csv')

    gt_csv_body = pd.read_csv(gt_csv_path)
    gt_csv_body = gt_csv_body[1:]
    gt_csv_body_column_names = gt_csv_body.columns.values.tolist()
    gt_csv_body_column_names.sort()
    gt_csv_body_column_names.pop()
    gt_csv_body_column_names.insert(0, 'frame')
    gt_csv_body = gt_csv_body[gt_csv_body_column_names]
    gt_csv_body = gt_csv_body.to_numpy()

    output_csv_body = pd.read_csv(output_csv_path)
    output_csv_body = output_csv_body[1:]
    output_csv_body = output_csv_body[gt_csv_body_column_names]
    output_csv_body = output_csv_body.to_numpy()
    results = []

    for idx in range(gt_csv_body.shape[0]):
        frame, *items = gt_csv_body[idx]
        
        is_correct = False
        for idx in range(output_csv_body.shape[0]):
            # print(int(output_csv_body[idx, 0]))
            if frame - 5 <= output_csv_body[idx, 0] <= frame + 5:
                # print("%d Frame - Output: ", output_csv_body[idx, 1:], ", GT: ", items)
                if (output_csv_body[idx, 1:] == items).all():
                    results.append(1)
                    is_correct = True
                    break

        if not is_correct:
            results.append(0)

    results = np.mean(results)

    result_str = "Target %s: Time: %dsec, Accuracy %.1f%%" % (target, elapsed_time, results * 100)
    print(result_str)
    with open('result.txt', 'a') as f:
        f.write(result_str + '\n')