import json

with open("embeddings/meta.json", encoding="utf-8") as f:
    meta = json.load(f)

print("Total chunks:", len(meta))
print("First 5 entries:\n")

for i in range(5):
    print(meta[i])
