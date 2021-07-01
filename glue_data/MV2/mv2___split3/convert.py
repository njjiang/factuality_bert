import pandas as pd
for x in ["test.jsonl", "train.jsonl", "dev.jsonl"]:
    df = pd.read_json(x)
    df.to_json(x, orient="records", lines=True)
