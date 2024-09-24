# %%
from pathlib import Path
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


results_dir = Path("/Users/vigji/Desktop/queries_results")
images_dir = Path("/Users/vigji/Desktop/queries_results/thumbnails")
images_dir.mkdir(parents=True, exist_ok=True)

all_results_files = sorted(results_dir.glob("results_*.json"))

# remove files that have already been downloaded, so there is at least a thumbnail:

all_results_files = [f for f in all_results_files if len(list((images_dir).glob(f"{f.stem}_*.png"))) == 0]

assert len(all_results_files) < 654
for results_file in tqdm(list(all_results_files)):
    with open(results_file, "r") as f:
        results = json.load(f)

    if "lens_results" in results:
        results = results["lens_results"]
    else:
        results = results["visual_matches"]
    
    # Keep track of number of failed downloads for each file, over total number of results:
    num_failed_downloads = 0
    for i, result in enumerate(results):
        thumbnail_url = result["thumbnail"]
        try:
            # open session with with to make sure we do not leave open sockets:
            with requests.Session() as session:
                retry = Retry(connect=10, backoff_factor=0.5)
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                thumbnail_response = session.get(thumbnail_url)

        except ConnectionError:
            print(f"ConnectionError for {thumbnail_url}")
            num_failed_downloads += 1
            continue
        if thumbnail_response.status_code == 200:
            thumbnail_data = thumbnail_response.content
            thumbnail_path = images_dir / f"{results_file.stem}_{i:03d}.png"
            with open(thumbnail_path, "wb") as f:
                f.write(thumbnail_data)
        else:
            print(f"Failed to download thumbnail for {thumbnail_url}")
            num_failed_downloads += 1

    print(f"Number of failed downloads for {results_file.stem}: {num_failed_downloads}")


# %%
[len(list((images_dir).glob(f"{results_file.stem}_*.png"))) for results_file in all_results_files[:3]]
# %%
l = [((images_dir / f.stem)) for f in all_results_files[:3]]# %%

# %%
list(l[0].glob("*"))    
# %%
