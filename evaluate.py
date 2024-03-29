import os
import pandas as pd
import numpy as np
import time

target_list = ['4p1b_01A2', '5p2b_01A1', '5p4b_01A2', '5p5b_03A1', '7p3b_02M']

for i in range(len(target_list)):
    target = target_list[i]

    video_path = os.path.join('..', 'videos', target, target + '.m4v')
    sample_csv_path = os.path.join('..', 'videos', target, target + '.csv')

    if not (os.path.exists(video_path) and os.path.exists(sample_csv_path)):
        print('Missing video or ground truth csv for target {}'.format(target))
        continue

    begin = time.time()
    os.system('python solution %s %s' % (video_path, sample_csv_path))
    end = time.time()
    
    elapsed_time = end - begin

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
                results.append((output_csv_body[idx, 1:] == items).astype(np.int32))
                is_correct = True
                break

        if not is_correct:
            results.append(np.zeros_like(output_csv_body[0, 1:]))

    results = np.mean(results)

    result_str = "Target %s: Time: %dsec, Accuracy %.1f%%" % (target, elapsed_time, results * 100)
    print(result_str)
    with open('result.txt', 'a') as f:
        f.write(result_str + '\n')