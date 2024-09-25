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

Even though enlarging the space of the training data is good, we want to try to be faithful to Eco's sense of beauty: we do not want to include images that are clearly different from the ones in the training set. 
To fix this, we can use a pre-trained CLIP model to compare the scraped images to the training set, and discard the scraped images that are too different. An alternative strategy is to use a pre-trained CNN like Resnet or VGG, cut the final classification layer, and use the penultimate layer as the image feature space. From there we can compute the cosine similarity between the training set and the scraped images, and discard the scraped images that are too different using an arbitrary threshold.

## Data inspection
