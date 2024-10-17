from serpapi import GoogleSearch
from pathlib import Path
import json
import requests


# video_results = results["video_results"]
results_dir = Path("/Users/vigji/My Drive/eco-beauty/queries_results")
results_list = []

serp = "scrapingdog"  #  "serpapi", "scrapingdog"
api_file_dict = {"serpapi": "key_serpapi.txt", "scrapingdog": "key_scrapingdog6.txt"}

with open(results_dir / api_file_dict[serp], "r") as f:
    api_key = f.read().strip()


labels = ["beautiful"]  # "ugly"
for label in labels:
    for i in range(627):
        results_file = results_dir / f"results_{label}_{i:03d}.json"

        if results_file.exists():
            print(f"Skipping {results_file} because it already exists")
            continue
        # url: i, placed within 3 digits number and trailing zeros:
        img_url = f"https://raw.githubusercontent.com/vigji/temp-eco/refs/heads/main/raw/{label}/img{i:03d}.png"

        if serp == "serpapi":
            params = {"engine": "google_lens", "url": img_url, "api_key": api_key}
            print(params)
            search = GoogleSearch(params)
            results = search.get_dict()
            print(results)
        elif serp == "scrapingdog":
            serp_url = "https://api.scrapingdog.com/google_lens"
            params = {
                "api_key": api_key,
                "url": f"https://lens.google.com/uploadbyurl?url={img_url}",
            }

            response = requests.get(serp_url, params=params)

            if response.status_code == 200:
                results = response.json()
                print(results)
            else:
                print(f"Request failed with status code: {response.status_code}")
                results = None

        else:
            raise ValueError(f"Unknown search engine: {serp}")

        results_list.append(results)

        with open(results_file, "w") as f:
            json.dump(results, f)
