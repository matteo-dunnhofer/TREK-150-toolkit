from __future__ import absolute_import, print_function

import six
import os
import numpy as np
import shutil
import json
from pycocotools.mask import decode

from ..utils.ioutils import download, extract, download_egoexo4d, extract_frames_cv2, save_ann_png, DAVIS_PALETTE

class VISTA(object):
    """The VISTA <http://machinelearning.uniud.it/datasets/vista/> Benchmark.

    Publication:
        ``Is Tracking really more challenging in First Person Egocentric Vision?``,
        Matteo Dunnhofer, Zaira Manigrasso, and Christian Micheloni, ICCV 2025.

    """

    def __init__(self, root_dir, split='test', mode='lt', anno_type='box', download=False):
        super(VISTA, self).__init__()

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.anno_type = anno_type
        self.just_first_anno = False

        if download:
            self._download(self.root_dir)
        
        with open(os.path.join(self.root_dir, 'annotations', f'{split}_{mode}_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        self.seq_names = list(self.annotations.keys())


    def __getitem__(self, index):
        """
        Args:
            index (integer or string): Index or name of a sequence.
            pov (string): 'fpv' or 'tpv' for first-person or third-person view.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        seq_name = self.seq_names[index]
        take = self.annotations[seq_name]['take']
        fpv_camera_name = self.annotations[seq_name][f'fpv_camera_name']
        tpv_camera_name = self.annotations[seq_name][f'tpv_camera_name']

        img_files_fpv = [os.path.join(self.root_dir, 'frames', take, fpv_camera_name, f'{fi}.jpg') for fi in list(self.annotations[seq_name]['frame_annotations'].keys())]
        img_files_tpv = [os.path.join(self.root_dir, 'frames', take, tpv_camera_name, f'{fi}.jpg') for fi in list(self.annotations[seq_name]['frame_annotations'].keys())]
        
        if self.just_first_anno:
            num_anno = 1
        else:
            num_anno = len(img_files_fpv)

        if self.anno_type == 'box':
            anno_fpv = np.zeros((num_anno, 4), dtype=np.float32) + np.nan
            anno_tpv = np.zeros((num_anno, 4), dtype=np.float32) + np.nan
        else:
            first_key = list(self.annotations[seq_name]['frame_annotations'].keys())[0]
            first_mask = decode(self.annotations[seq_name]['frame_annotations'][first_key]['fpv']['mask'])
            anno_fpv = np.zeros((num_anno, first_mask.shape[0], first_mask.shape[1]), dtype=np.uint8) + np.nan

            first_key = list(self.annotations[seq_name]['frame_annotations'].keys())[0]
            first_mask = decode(self.annotations[seq_name]['frame_annotations'][first_key]['tpv']['mask'])
            anno_tpv = np.zeros((num_anno, first_mask.shape[0], first_mask.shape[1]), dtype=np.uint8) + np.nan

        # load annotations
        frame_idxs = list(self.annotations[seq_name]['frame_annotations'].keys())
        for i in range(num_anno):
            fi = frame_idxs[i]
            if len(self.annotations[seq_name]['frame_annotations'][fi]) > 0:
                if self.anno_type == 'box':
                    anno_fpv[i, :] = np.array(self.annotations[seq_name]['frame_annotations'][fi]['fpv']['box'])
                    anno_tpv[i, :] = np.array(self.annotations[seq_name]['frame_annotations'][fi]['tpv']['box'])
                else:
                    anno_fpv[i, :, :] = decode(self.annotations[seq_name]['frame_annotations'][fi]['fpv']['mask'])
                    anno_tpv[i, :, :] = decode(self.annotations[seq_name]['frame_annotations'][fi]['tpv']['mask'])

        if self.just_first_anno:
            assert len(img_files_fpv) == len(img_files_tpv)
        else:
            assert len(img_files_fpv) == len(anno_fpv) == len(img_files_tpv) == len(anno_tpv)
        

        return img_files_fpv, anno_fpv, img_files_tpv, anno_tpv, frame_idxs

    def __len__(self):
        return len(self.seq_names)
        
    def _download(self, root_dir):

        print('*** Downloading and processing VISTA frames from EgoExo4D. This process might take a while...')

        url_fmt_vista = 'https://machinelearning.uniud.it/datasets/vista/VISTA-annotations.zip'

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
    
        # Download list of sequences
        anno_file = os.path.join(root_dir, 'annotations', f'{self.split}_{self.mode}_annotations.json')
        if not os.path.exists(anno_file):
            # Download the archive containing the annotations
            anno_zip_file = os.path.join(root_dir, 'VISTA-annotations.zip')
            if not os.path.exists(anno_zip_file):
                download(url_fmt_vista, anno_zip_file)
                extract(anno_zip_file, root_dir)

        assert os.path.exists(anno_file), f"Annotations file {anno_file} does not exist. Check the download process."

        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)

        seq_names = list(self.annotations.keys())
        n_sequences = len(seq_names)
        take_names = [self.annotations[seq_name]['take'] for seq_name in seq_names]
        take_names = np.unique(take_names)

        # Download sequence annotations and frames
        tmp_dir = os.path.join(root_dir, 'tmp')

        seq_idx = 0
        for take in take_names:
            take_dir = os.path.join(root_dir, 'frames', take)

            for seq_name in seq_names:
                if take == self.annotations[seq_name]['take']:
                    print(f'*** *** [{seq_idx+1}/{n_sequences}] Processing sequence {seq_name} with take {take}...')
                    frame_idxs = list(self.annotations[seq_name]['frame_annotations'].keys())

                    for pov in ['fpv', 'tpv']:
                        camera_name = self.annotations[seq_name][f'{pov}_camera_name']

                        #print([not os.path.exists(os.path.join(take_dir, camera_name, f'{fi}.jpg')) for fi in frame_idxs])
                        if any([not os.path.exists(os.path.join(take_dir, camera_name, f'{fi}.jpg')) for fi in frame_idxs]):

                            if not os.path.exists(tmp_dir):
                                os.makedirs(tmp_dir)
                            
                            if not os.path.exists(os.path.join(take_dir, camera_name)):
                                os.makedirs(os.path.join(take_dir, camera_name))
                            
                            if not os.path.exists(os.path.join(tmp_dir, 'takes', f'{take}')):
                                # Download the MP4 video file
                                download_egoexo4d(self.annotations[seq_name]['annotation_id'], tmp_dir)

                            print(f'*** *** *** *** Extracting {pov} frames ...')
                            extract_frames_cv2(os.path.join(tmp_dir, 'takes', f'{take}', 'frame_aligned_videos', f'{camera_name}.mp4'), os.path.join(take_dir, f'{camera_name}'), frame_idxs, resolution=720)
                            
                            assert all([os.path.exists(os.path.join(take_dir, camera_name, f'{fi}.jpg')) for fi in frame_idxs]), f"Not all frames were extracted for {take} - {camera_name}. Check the download process."

                    print(f'*** *** *** *** Sequence {seq_name} OK!')
                    seq_idx += 1

            if os.path.exists(os.path.join(tmp_dir, 'takes', f'{take}')):
                shutil.rmtree(os.path.join(tmp_dir, 'takes', f'{take}'), ignore_errors=True)
                print(f'*** *** Cleaning take {take} temporary files...')
            
        if os.path.exists(tmp_dir): 
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f'*** *** Cleaning temporary directory...')

        print('***')
        print('***')
        print('***')
        print('*** VISTA dataset is ready!')

    def export_as_davis(self):
        """
        Exports the VISTA dataset into a DAVIS-like directory structure.
        This method copies image frames and annotation masks for each sequence and point of view (FPV and TPV)
        into a DAVIS-style folder hierarchy under the dataset root. Images are placed in 'JPEGImages' and masks
        in 'Annotations', organized by sequence name. Masks are decoded and saved as PNG files using the DAVIS palette.
        """

        save_folder = os.path.join(self.root, 'davis_like')

        print(f'*** Copying VISTA into a DAVIS-like format at {save_folder}. This process might take a while...')

        save_folder_fpv = os.path.join(save_folder, 'fpv')
        save_folder_tpv = os.path.join(save_folder, 'tpv')

        for folder in ['JPEGImages', 'Annotations']:
            for pov in ['fpv', 'tpv']:
                if not os.path.exists(os.path.join(save_folder, pov, folder)):
                    os.makedirs(os.path.join(save_folder, pov, folder))

        n_sequences = len(self.seq_names)

        for seq_idx, seq_name in enumerate(self.seq_names):

            print(f'*** *** [{seq_idx+1}/{n_sequences}] Processing sequence {seq_name}...')

            take = self.annotations[seq_name]['take']
            fpv_camera_name = self.annotations[seq_name][f'fpv_camera_name']
            tpv_camera_name = self.annotations[seq_name][f'tpv_camera_name']

            img_files_fpv = [os.path.join(self.root_dir, 'frames', take, fpv_camera_name, f'{fi}.jpg') for fi in list(self.annotations[seq_name]['frame_annotations'].keys())]
            img_files_tpv = [os.path.join(self.root_dir, 'frames', take, tpv_camera_name, f'{fi}.jpg') for fi in list(self.annotations[seq_name]['frame_annotations'].keys())]
            
            for folder in ['JPEGImages', 'Annotations']:
                for pov in ['fpv', 'tpv']:
                    if not os.path.exists(os.path.join(save_folder, pov, folder, seq_name)):
                        os.makedirs(os.path.join(save_folder, pov, folder, seq_name))

            for img_file in img_files_fpv:
                img_file_name = os.path.basedir(img_file)
                shutil.copy(img_file, os.path.join(save_folder, 'fpv', 'JPEGImages', seq_name, img_file_name))

            for img_file in img_files_tpv:
                img_file_name = os.path.basedir(img_file)
                shutil.copy(img_file, os.path.join(save_folder, 'tpv', 'JPEGImages', seq_name, img_file_name))  

      
            frame_idxs = list(self.annotations[seq_name]['frame_annotations'].keys())
            for i in range(len(frame_idxs)):
                fi = frame_idxs[i]
                if len(self.annotations[seq_name]['frame_annotations'][fi]) > 0:
                    mask_fpv = decode(self.annotations[seq_name]['frame_annotations'][fi]['fpv']['mask'])
                    save_ann_png(os.path.join(save_folder, 'fpv', 'Annotations', seq_name, f'{fi}.png'), mask_fpv, DAVIS_PALETTE)

                    mask_tpv = decode(self.annotations[seq_name]['frame_annotations'][fi]['tpv']['mask'])
                    save_ann_png(os.path.join(save_folder, 'tpv', 'Annotations', seq_name, f'{fi}.png'), mask_tpv, DAVIS_PALETTE)

            print(f'*** *** *** Sequence {seq_name} OK!')

        print('***')
        print('***')
        print('***')
        print('*** VISTA DAVIS-like dataset is ready!')



