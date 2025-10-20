from datasets import load_dataset

ds = load_dataset("qwedsacf/competition_math")
print(ds)
print(ds['train'][0])
