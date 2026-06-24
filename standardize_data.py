#!/usr/bin/env python3
"""
只标准化 extra_info 字段，保持其他字段不变
"""

import pandas as pd
import json
import os
from glob import glob
from tqdm import tqdm
import numpy as np

DATA_ROOT = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"

def safe_json_dumps(obj):
    """安全地将对象转换为JSON字符串"""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):
        obj = int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        obj = float(obj)
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except:
        return str(obj)

def standardize_extra_info_only(file_path):
    """只标准化 extra_info 字段"""
    try:
        df = pd.read_parquet(file_path)
        
        if len(df) == 0:
            return False, "empty"
        
        # 只处理 extra_info 列
        if 'extra_info' in df.columns:
            # 将 extra_info 转换为 JSON 字符串
            df['extra_info'] = df['extra_info'].apply(safe_json_dumps)
        
        # 保存（覆盖原文件）
        df.to_parquet(file_path, index=False)
        return True, "success"
        
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("标准化 extra_info 字段（保持其他字段不变）")
    print("=" * 80)
    
    # 查找所有 parquet 文件（排除备份）
    all_files = glob(f"{DATA_ROOT}/**/*.parquet", recursive=True)
    all_files = [f for f in all_files if not f.endswith('.backup')]
    
    print(f"找到 {len(all_files)} 个文件")
    
    success = 0
    failed = []
    
    for file_path in tqdm(all_files, desc="处理进度"):
        ok, status = standardize_extra_info_only(file_path)
        if ok:
            success += 1
        elif status != "empty":
            failed.append((file_path, status))
    
    print()
    print(f"成功: {success}")
    print(f"失败: {len(failed)}")
    
    if failed:
        for f, e in failed[:10]:
            print(f"  - {f}: {e}")

if __name__ == "__main__":
    main()
