from __future__ import absolute_import, print_function

import six
import os
import glob
import numpy as np
import shutil
import subprocess

from ..utils.ioutils import download, extract


class TREK150(object):
    """The TREK-150 <http://machinelearning.uniud.it/datasets/trek150/> Benchmark.

    Publication:
        ``Is First Person Vision Challenging for Object Tracking?``,
        Matteo Dunnhofer, Antonino Furnari, Giovanni Maria Farinella and Christian Micheloni, ICCVW 2021.

    """

    def __init__(self, root_dir, download=True):
        super(TREK150, self).__init__()

        self.root_dir = root_dir

        if download:
            self._download(self.root_dir)

        # sequence and annotation paths
        self.anno_files = sorted(glob.glob(
            os.path.join(self.root_dir, '*/groundtruth_rect.txt')))

        self.seq_names = [f.split('/')[-2] for f in self.anno_files]
        self.seq_dirs = [os.path.join(self.root_dir, n) for n in self.seq_names]

    def __getitem__(self, index):
        """
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))

        # load annotations
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)
        
    def _download(self, root_dir):

        print('Checking and downloading TREK-150. This process might take a while...')

        url_fmt_trek150 = 'https://machinelearning.uniud.it/datasets/trek150/TREK-150-annotations.zip'
        url_fmt_ek = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/'

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
    
        # Download list of sequences
        seqs_file = os.path.join(root_dir, 'TREK-150-annotations', 'sequences.txt')
        if not os.path.exists(seqs_file):
            # Download the archive containing the annotations
            anno_zip_file = os.path.join(root_dir, 'TREK-150-annotations.zip')
            if not os.path.exists(anno_zip_file):
                download(url_fmt_trek150, anno_zip_file)
                extract(anno_zip_file, root_dir)

        assert os.path.exists(seqs_file)

        seq_names = np.genfromtxt(seqs_file, delimiter='\n', dtype="str")

        # Download sequence annotations and frames
        for i, seq_name in enumerate(seq_names):
            seq_dir = os.path.join(root_dir, seq_name)

            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)

            print(f'Processing video sequence {seq_name} [{i+1}/{len(seq_names)}]')

            if (not os.path.exists(os.path.join(seq_dir, 'groundtruth_rect.txt'))) or \
                (not os.path.exists(os.path.join(seq_dir, 'frames.txt'))) or \
               (not os.path.exists(os.path.join(seq_dir, 'attributes.txt'))) or \
               (not os.path.exists(os.path.join(seq_dir, 'action_target.txt'))):

                # Extract annotations
                zip_file = os.path.join(root_dir, 'TREK-150-annotations', seq_name + '.zip')
        
                print('\n\tExtracting annotation to %s...' % root_dir)
                extract(zip_file, seq_dir)
            else:
                print('\tAnnotation already extracted!')

            # Download frames from EPIC-Kitchens

            # Get partecipant and video idxs
            id_split = seq_name.split('-')
            participant_idx = id_split[0]
            video_idx = id_split[1]

            # Build EK-55 video url
            url_ek = os.path.join(url_fmt_ek, participant_idx, video_idx + '.MP4')
            mp4_file = os.path.join(root_dir, video_idx + '.MP4')
            mp4_frames_dir = os.path.join(root_dir, video_idx)

        
            frame_idxs = np.loadtxt(os.path.join(seq_dir, 'frames.txt'), delimiter='\n', dtype=np.uint64)

            seq_img_dir = os.path.join(seq_dir, 'img')

            # Copying images if are not already copied
            if not os.path.isdir(seq_img_dir):
                os.makedirs(seq_img_dir)

            # If frames have not been already processed
            if len(os.listdir(seq_img_dir)) != frame_idxs.shape[0]:

                # Download MP4 video if not alrady done
                if not os.path.exists(mp4_file):
                    print('\n\tDownloading EK video to %s...' % mp4_file)
                    download(url_ek, mp4_file)
                else:
                    print(f'\tEK video {video_idx} already dowloaded!')

                # Extract frames from the MP4 videos using ffmpeg
                if not os.path.isdir(mp4_frames_dir):
                    os.makedirs(mp4_frames_dir)
                    subprocess.call(['ffmpeg', '-i', mp4_file, '-r', '60', mp4_frames_dir + '/frame_%010d.jpg'],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
                    print('\tEK video frames extracted!')
                else:
                    print('\tEK video frames already extracted!')

                for fi in frame_idxs:
                    fi_file = f'frame_{fi:010d}.jpg'
                    shutil.copy(os.path.join(mp4_frames_dir, fi_file), os.path.join(seq_img_dir, fi_file))
            else:
                print('\tSequence frames already copied!')

        # Remove the downloaded MP4 files
        mp4_files = glob.glob(os.path.join(root_dir, '*.MP4'))
        for mp4_file in mp4_files:
            os.remove(mp4_file)

        # Remove eventually downloaded tmp files
        tmp_files = glob.glob(os.path.join(root_dir, '*.tmp'))
        for tmp_file in tmp_files:
            os.remove(tmp_file)

        # Remove the dirs containing the raw frames
        for seq_name in seq_names:
            id_split = seq_name.split('-')
            #participant_idx = id_split[0]
            video_idx = id_split[1]

            shutil.rmtree(os.path.join(root_dir, video_idx), ignore_errors=True)

