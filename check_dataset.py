import os

data_dir = "Data_set"
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} not found!")
    exit(1)

for split in ["train", "val", "test"]:
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        print(f"Split {split} not found!")
        continue
    print(f"\n--- Split: {split} ---")
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    total = 0
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        print(f"  {cls}: {count}")
        total += count
    print(f"Total for {split}: {total}")
