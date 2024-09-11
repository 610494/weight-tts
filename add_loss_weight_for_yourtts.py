import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def find_max_score(data):
    # 提取 test_scores 列表中的第一個數字（得分）
    scores = [score[0] for score in data['test_scores']]
    
    # 找出最大值
    max_score = max(scores)
    
    return max_score

if __name__=='__main__':
    
    # json_path = '/mnt/md1/user_wago/data/mandarin_drama/mandarin_drama_95hr_sub_select_len_free.json'
    # csv_path = '/mnt/md1/user_wago/data/mandarin_drama/mandarin_drama_95hr_sub_select_text_with_pinyin_dict.csv'
    # output_path = '/mnt/md1/user_wago/data/mandarin_drama/mandarin_drama_95hr_sub_select_text_with_pinyin_dict_with_loss_weight.csv'
    
    json_path = '/mnt/md1/user_wago/data/mandarin_drama/mandarin_drama_95hr_sub_select_len_free_sub_34hr.json'
    csv_path = '/mnt/md1/user_wago/MOS/csv_test/mandarin_drama_95hr_sub_select_subset_34hr.csv'
    
    alpha = -1
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    if alpha == -1:
        alpha = find_max_score(json_data)
    print(f'alpha: {alpha}')
    test_scores = json_data['test_scores']

    output_path = csv_path.replace('.csv', f'_alpha_{alpha}.csv')
    
    df = pd.read_csv(csv_path)

    # 迭代 test_scores 中的每個項目
    for score, path in tqdm(test_scores):
        # 取得路徑的 basename
        # basename = os.path.basename(path)
        
        mask = df['filename'] == path
        
        # print(f'deep_svdd_score: {score}, weight: {1 - np.clip(1 / alpha * score, 0, 1) }')
        df.loc[mask, 'deep_svdd_score'] = score
        df.loc[mask, 'weight'] = 1 - np.clip(1 / alpha * score, 0, 1) 
        

    # 將結果寫到新的 CSV 檔案
    df.to_csv(output_path, index=False)
