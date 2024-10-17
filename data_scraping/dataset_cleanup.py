# Resizes and pads images to a desired size

from pathlib import Path
from PIL import Image


def resize_pad(img, desired_size=500):
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)  # resize the input image

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )

    return new_im


master_folder = Path("/Users/luigipetrucco/Desktop/ecocide")
desired_size = 500

for group in ["belli", "brutti"]:
    k = 0
    for image_path in (master_folder / "raw" / group).glob("*.png"):
        img = Image.open(image_path)
        padded_img = resize_pad(img, desired_size=desired_size)
        padded_img.save(
            str(master_folder / "cleaned" / group / "img{:03d}.png".format(k)),
            format="png",
        )
        k += 1
