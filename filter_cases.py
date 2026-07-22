import json, os

folder = "./InjecAgent/adaptive_attack_results"
prefix = "260705211435_GCG_Llama-3.1-8B-Instruct_InstructionalPrevention_prefix_base_subset_20"

for suffix in ["dh_data.json", "ds_data.json"]:
    path = os.path.join(folder, f"{prefix}_{suffix}")
    with open(path, 'r') as f:
        cases = json.load(f)
    filtered = [c for c in cases if 'Adv String' in c]
    with open(path, 'w') as f:
        json.dump(filtered, f, indent=4)
    print(f"{path}: kept {len(filtered)} of {len(cases)} cases")