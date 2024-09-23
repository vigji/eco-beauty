from serpapi import GoogleSearch
from pathlib import Path
import json

# video_results = results["video_results"]
results_dir = Path("/Users/vigji/Desktop/queries_results")
results_list = []
key_file = Path("/Users/vigji/Desktop/queries_results/key.txt")
with open(key_file, "r") as f:
    api_key = f.read().strip()

labels = ["ugly"] #, "beautiful"]
for label in labels:
    for i in range(100):
        results_file = results_dir / f"results_{label}_{i:03d}.json"
        if results_file.exists():
            print(f"Skipping {results_file} because it already exists")
            continue
        # url: i, placed within 3 digits number and trailing zeros:
        url = f"https://raw.githubusercontent.com/vigji/temp-eco/refs/heads/main/raw/{label}/img{i:03d}.png"
        print(url)
        params = {
            "engine": "google_lens",
            "url": url,
            "api_key": api_key
        }
        print(params)
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)

        results_list.append(results)

        with open(results_file, "w") as f:
            json.dump(results, f)


    