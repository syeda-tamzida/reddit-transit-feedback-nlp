from huggingface_hub import list_repo_files

files = list_repo_files("fddemarco/pushshift-reddit-comments", repo_type="dataset")
months = sorted(f[8:15] for f in files if f.startswith("data/RC_"))

print("Number of months:", len(months))
print("First month:", months[0])
print("Last month:", months[-1])
print(months)
