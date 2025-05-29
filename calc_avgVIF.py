import os
import pandas as pd
from collections import defaultdict
import math
import numpy as np


BASE_DIR = "/hopper/groups/enkavilab/data/ds004636/derivatives"
RT_TYPES = ["rt_centered", "rt_duration", "rt_duration_only", "rt_uncentered"]
# RT_TYPES=["rt_centered"]
OUTPUT_DIR = "/hopper/groups/enkavilab/users/ibrayyilmaz/rt_data_analysis/rt_data_analysis/main_analysis_code/avgVIF"  # saves in current working directory

def get_task_dirs(base_dir):
    return [d for d in os.listdir(base_dir) if d.endswith("_glm") and os.path.isdir(os.path.join(base_dir, d))]

def collect_vif_data(task_dir, task_name):
    task_path = os.path.join(BASE_DIR, task_dir)
    vif_data = {rt: defaultdict(list) for rt in RT_TYPES}

    for sub in os.listdir(task_path):
        sub_path = os.path.join(task_path, sub)
        if not os.path.isdir(sub_path): continue

        for rt in RT_TYPES:
            rt_path = os.path.join(sub_path, rt)
            if not os.path.isdir(rt_path): continue

            for file in os.listdir(rt_path):
                if file.startswith("VIF_") and file.endswith(".csv"):
                    file_path = os.path.join(rt_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        key_col = "regressor" if "regressor" in df.columns else "contrast"
                        for _, row in df.iterrows():
                            # for each rt_type and regressor per task, add to array a subject VIF value.
                            vif_data[rt][row[key_col]].append(float(row["VIF"]))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return task_name, vif_data

# updated to remove all nan and infinities
def average_vifs(vif_dict):
    for k, v in vif_dict.items():
        v_filtered = v[np.isfinite(v)]
        v_avg=sum(v_i for v_i in v_filtered)/len(v_i for v_i in v_filtered)
        v_std=np.std(v_filtered)
        return {k : [v_avg,v_std]}
        
        
    

def main():
    task_dirs = get_task_dirs(BASE_DIR)
    all_vif = {rt: {} for rt in RT_TYPES}  # rt -> task -> {regressor/contrast -> [VIFs]}

    for task_dir in task_dirs:
        task_name = task_dir.replace("_glm", "")
        task_name, task_vif_data = collect_vif_data(task_dir, task_name)
        
        for rt in RT_TYPES:
            averaged = average_vifs(task_vif_data[rt])
            all_vif[rt][task_name] = averaged

    # Write output files
    for rt in RT_TYPES:
        df = pd.DataFrame(all_vif[rt]) # convert to DataFrame: rows=regressors, columns=tasks
        out_name = f"avgVIF_pertask_{rt}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path, index=True)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()