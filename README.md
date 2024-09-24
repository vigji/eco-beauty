# Eco, on beauty and ugliness

Art history & machine learning

## Data scraping

The biggest risk for this model is to overfit implicit watermarks of the book the images are from,
like color histograms, compression patterns, etc.

To avoid the issue, after scraping all images, we use SERP APIs to query the images in Google Lens,
and scrape the thumbnails of the results. Thumbnails are low quality but overall big enough (given that we rescale them to 256x256 anyway).
This comes with the advantage of getting very different images in terms of vignetting, background, frame, lighting, etc. For some images,
we also could potentially get similar works from the same author/period, different views of the same artworks...this is actually also good!

The code for scraping the images is in `data_scraping/`. `google_images_scraping.py` contains code for making the SERP calls, and `download_thumbnails.py` contains code for downloading the thumbnails.

## Data inspection
