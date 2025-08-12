from __future__ import absolute_import, division

import wget
import os
import shutil
import zipfile
import subprocess
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download(url, filename):
    r"""Download file from the internet.
    
    Args:
        url (string): URL of the internet file.
        filename (string): Path to store the downloaded file.
    """
    return wget.download(url, out=filename)

def download_egoexo4d(uid, root_dir, s3_profile='egoexo4d'):
    command = ['egoexo', '-o', root_dir, '--parts', 'takes', '--uids', uid, "--s3_profile", s3_profile, "--release", "v2"]
    print('Command to execute:', command)
    output = subprocess.run(command, input=b'yes\n')
    print(output.stdout)


def extract(filename, extract_dir):
    r"""Extract zip file.
    
    Args:
        filename (string): Path of the zip file.
        extract_dir (string): Directory to store the extracted results.
    """
    if os.path.splitext(filename)[1] == '.zip':
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.'.format(
            os.path.splitext(filename)[1]), filename)


def compress_file(dirname, save_file):
    """Compress a folder to a zip file.
    
    Arguments:
        dirname {string} -- Directory of all files to be compressed.
        save_file {string} -- Path to store the zip file.
    """
    shutil.make_archive(save_file, 'zip', dirname)


def extract_frames_cv2(clip_path, frames_save_dir, frame_idxs, resolution=-1):
    # Open the video file
    video = cv2.VideoCapture(clip_path)

    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through frames and save them
    ret, frame = video.read()
    frame_number = 0
    valid_frames = 0

    while ret:
        if str(frame_number) in frame_idxs:
            output_path = os.path.join(frames_save_dir, f'{frame_number}.jpg')
            if resolution > 0:
                frame = resize_image(frame, resolution)
            cv2.imwrite(output_path, frame)
            valid_frames += 1

        ret, frame = video.read()
        frame_number += 1
    
    # Release video object
    video.release()
    
    #print("Frames extracted successfully.")


def resize_image(image, size):
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Determine the scaling factor to resize the image
    if height < width:
        # If height is smaller, resize based on height = 720
        scale_factor = size / height
    else:
        # Otherwise, resize based on width = 720
        scale_factor = size / width

    # Calculate the new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)
