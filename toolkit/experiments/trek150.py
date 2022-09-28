from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import itertools
import json
from PIL import Image
import time
import copy
import glob

from ..datasets import TREK150
from ..utils.metrics import rect_iou, center_error, normalized_center_error
from ..utils.viz import show_frame

class ExperimentTREK150(object):
    r"""Experiment pipeline and evaluation toolkit for the TREK-150 dataset.

    Args:
        root_dir (string): Root directory of OTB dataset.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``. Default is ``2015``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, result_dir='./results', report_dir='./reports'):
        super(ExperimentTREK150, self).__init__()
        self.root_dir = root_dir
        self.dataset = TREK150(root_dir, download=True)
        self.result_dir = os.path.join(result_dir, 'TREK-150')
        self.report_dir = os.path.join(report_dir, 'TREK-150')
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51
        self.nbins_gsr = 51

        self.fps = 60

    def run(self, tracker, protocol='ope', visualize=False):
        if protocol == 'ope':
            self.run_ope(tracker, visualize=visualize)
        elif protocol == 'mse':
            self.run_mse(tracker, visualize=visualize)
        elif protocol == 'rte':
            self.run_rte(tracker, visualize=visualize)
        elif protocol == 'hoi':
            self.run_hoi(tracker, visualize=visualize)

    def run_ope(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'ope', '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)

    def run_mse(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))
            
            anchors = np.loadtxt(os.path.join(self.root_dir, seq_name, 'anchors.txt'), delimiter=',')
            if len(anchors.shape) == 1:
                anchors = np.array([anchors])

            for i in range(anchors.shape[0]):
                anchor = int(anchors[i,0])
                direction = int(anchors[i,1])

                dir_str = 'forward' if direction < 1 else 'backward'
                print(f'Anchor {anchor} - Direction {dir_str}')

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, 'mse', f'{seq_name}-anchor-{anchor}.txt')
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                if direction < 1:
                    img_files_, anno_ = img_files[anchor:], anno[anchor:]
                else:
                    img_files_, anno_ = img_files[:anchor+1], anno[:anchor+1]
                    img_files_, anno_ = list(reversed(img_files_)), anno_[::-1]

                # tracking loop
                boxes, _ = tracker.track(
                    img_files_, anno_[0, :], visualize=visualize)
                assert len(boxes) == len(anno_)

                # record results
                self._record(record_file, boxes)

    def run_rte(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'rte', '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            boxes = []

            total_time = 0.0
            offset = 0

            frame_time = 0.0

            # tracking loop
            for f, img_file in enumerate(img_files):

                frame = Image.open(img_file)
                if not frame.mode == 'RGB':
                    frame = frame.convert('RGB')
            
                frame_time += (1.0 / self.fps)

                if f == 0:
                    # during initialization frames
                    start_time = time.time()

                    tracker.init(frame, anno[f, :])

                    last_time = time.time() - start_time

                    total_time = max(1.0 / self.fps, last_time) # last_time

                    boxes.append(anno[f, :])

                    offset = 0 

                    last_bbox = anno[0, :]
                    bbox = anno[0, :]
                else:
                    current = offset + int(np.floor(total_time * self.fps))

                    if f == current:

                        last_bbox = copy.deepcopy(bbox)

                        start_time = time.time()

                        bbox = tracker.update(frame)

                        last_time = time.time() - start_time

                        if total_time + last_time < frame_time:
                            boxes.append(bbox)
                            total_time += last_time + (frame_time - total_time + last_time)
                        else:
                            total_time += last_time
                            boxes.append(last_bbox)
                    else:
                        boxes.append(last_bbox)
                
                if visualize:
                    if len(boxes[-1]) == 4:
                        show_frame(frame, boxes[-1])
                    else:
                        show_frame(frame)

            # record results
            self._record(record_file, boxes)

    def run_hoi(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))
            
            anchors = np.loadtxt(os.path.join(self.root_dir, seq_name, 'anchors_hoi.txt'), delimiter=',')
            if len(anchors.shape) == 1:
                anchors = np.array([anchors])

            for i in range(anchors.shape[0]):
                start_idx = int(anchors[i,0])
                end_idx = int(anchors[i,1])
                inter_idx = int(anchors[i,2])

                if inter_idx == 0:
                    dir_str = 'LHI' 
                elif direction == 1:
                    dir_str = 'RHI' 
                else:
                    dir_str = 'BHI' 
                print(f'HOI starting at {start_idx} ending at {end_idx} - Type {dir_str}')

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, 'hoi', f'{seq_name}-hoi-{start_idx}-{end_idx}-{inter_idx}.txt')
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                img_files_, anno_ = img_files[start_idx:end_idx+1], anno[start_idx:end_idx+1]

                # tracking loop
                boxes, _ = tracker.track(
                    img_files_, anno_[0, :], visualize=visualize)
                assert len(boxes) == len(anno_)

                # record results
                self._record(record_file, boxes)

    def report(self, tracker_names, protocol='ope'):
        if protocol == 'ope':
            self.report_ope(tracker_names)
        elif protocol == 'mse':
            self.report_mse(tracker_names)
        elif protocol == 'rte':
            self.report_ope(tracker_names, realtime=True)
        elif protocol == 'hoi':
            self.report_hoi(tracker_names)
        
    def report_ope(self, tracker_names, realtime=False):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        if realtime:
            report_file = os.path.join(report_dir, 'performance-rte.json')
        else:
            report_file = os.path.join(report_dir, 'performance-ope.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]

                if realtime:
                    record_file = os.path.join(
                            self.result_dir, name, 'rte', '%s.txt' % seq_name)
                else:
                    record_file = os.path.join(
                            self.result_dir, name, 'ope', '%s.txt' % seq_name)

                boxes = np.loadtxt(record_file, delimiter=',')

                boxes[0] = anno[0, :]
                assert len(boxes) == len(anno)

                #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                ious, norm_center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], norm_prec_curve[s] = self._calc_curves(ious, norm_center_errors)
                gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

                # calculate average tracking speed
                if not realtime:
                    time_file = os.path.join(
                        self.result_dir, name, 'ope', 'times', '%s_time.txt' % seq_name)
                    if os.path.isfile(time_file):
                        times = np.loadtxt(time_file)
                        times = times[times > 0]
                        if len(times) > 0:
                            speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'normalized_precision_curve': norm_prec_curve[s].tolist(),
                    'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'normalized_precision_score': np.mean(norm_prec_curve[s]),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            norm_prec_curve = np.mean(norm_prec_curve, axis=0)
            gen_succ_rob_curve = np.mean(gen_succ_rob_curve, axis=0)
            succ_score = np.mean(succ_curve)
            norm_prec_score = np.mean(norm_prec_curve)
            gen_succ_rob_score = np.mean(gen_succ_rob_curve)
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
                'success_score': succ_score,
                'normalized_precision_score': norm_prec_score,
                'generalized_success_robustness_score': gen_succ_rob_score,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        if not realtime:
            self.plot_curves(tracker_names)

        return performance

    def report_mse(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, 'performance-mse.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)

            overall_succ_score = 0.0
            overall_norm_prec_score = 0.0
            overall_gen_succ_rob_score = 0.0
            overall_seq_length = 0

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]

                anchors = np.loadtxt(os.path.join(self.root_dir, seq_name, 'anchors.txt'), delimiter=',')
                if len(anchors.shape) == 1:
                    anchors = np.array([anchors])

                seq_succ_score = 0.0
                seq_norm_prec_score = 0.0
                seq_gen_succ_rob_score = 0.0
                valid_frames = 0

                seq_length = anno.shape[0]

                for i in range(anchors.shape[0]):
                    anchor = int(anchors[i,0])
                    direction = int(anchors[i,1])

                    record_file = os.path.join(
                            self.result_dir, name, 'mse', f'{seq_name}-anchor-{anchor}.txt')

                    boxes = np.loadtxt(record_file, delimiter=',')

                    if direction < 1:
                        anno_ = anno[anchor:]
                    else:
                        anno_ = anno[:anchor+1]
                        anno_ = anno_[::-1]

                    boxes[0] = anno_[0]
                    assert len(boxes) == len(anno_)

                    ious, norm_center_errors = self._calc_metrics(boxes, anno_)
                    succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                    gen_succ_rob_curve = self._calc_curves_robustness(ious)

                    succ_score = np.mean(succ_curve) * anno_.shape[0]
                    norm_prec_score = np.mean(norm_prec_curve) * anno_.shape[0]
                    gen_succ_rob_score = np.mean(gen_succ_rob_curve) * anno_.shape[0]

                    seq_succ_score += succ_score
                    seq_norm_prec_score += norm_prec_score
                    seq_gen_succ_rob_score += gen_succ_rob_score

                    valid_frames += anno_.shape[0]
                    
                seq_succ_score /= valid_frames
                seq_norm_prec_score /= valid_frames
                seq_gen_succ_rob_score /= valid_frames

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_score': seq_succ_score,
                    'normalized_precision_score': seq_norm_prec_score,
                    'generalized_success_robustness_score': seq_gen_succ_rob_score}})

                overall_succ_score += (seq_succ_score * seq_length)
                overall_norm_prec_score += (seq_norm_prec_score * seq_length)
                overall_gen_succ_rob_score += (seq_gen_succ_rob_score * seq_length)

                overall_seq_length += seq_length

            overall_succ_score /= overall_seq_length
            overall_norm_prec_score /= overall_seq_length
            overall_gen_succ_rob_score /= overall_seq_length

            # store overall performance
            performance[name]['overall'].update({
                'success_score': overall_succ_score,
                'normalized_precision_score': overall_norm_prec_score,
                'generalized_success_robustness_score': overall_gen_succ_rob_score})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        return performance

    def report_hoi(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, 'performance-hoi.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)

            overall_succ_score = 0.0
            overall_norm_prec_score = 0.0
            overall_gen_succ_rob_score = 0.0
            overall_seq_length = 0

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]

                anchors = np.loadtxt(os.path.join(self.root_dir, seq_name, 'anchors_hoi.txt'), delimiter=',')
                if len(anchors.shape) == 1:
                    anchors = np.array([anchors])

                seq_succ_score = 0.0
                seq_norm_prec_score = 0.0
                seq_gen_succ_rob_score = 0.0
                valid_frames = 0

                seq_length = anno.shape[0]

                for i in range(anchors.shape[0]):
                    start_idx = int(anchors[i,0])
                    end_idx = int(anchors[i,1])
                    inter_idx = int(anchors[i,2])

                    record_file = os.path.join(
                            self.result_dir, name, 'hoi', f'{seq_name}-hoi-{start_idx}-{end_idx}-{inter_idx}.txt')

                    boxes = np.loadtxt(record_file, delimiter=',')

                    anno_ = anno[start_idx:end_idx+1]

                    boxes[0] = anno_[0]
                    assert len(boxes) == len(anno_)

                    ious, norm_center_errors = self._calc_metrics(boxes, anno_)
                    succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                    gen_succ_rob_curve = self._calc_curves_robustness(ious)

                    succ_score = np.mean(succ_curve) * anno_.shape[0]
                    norm_prec_score = np.mean(norm_prec_curve) * anno_.shape[0]
                    gen_succ_rob_score = np.mean(gen_succ_rob_curve) * anno_.shape[0]

                    seq_succ_score += succ_score
                    seq_norm_prec_score += norm_prec_score
                    seq_gen_succ_rob_score += gen_succ_rob_score

                    valid_frames += anno_.shape[0]
                    
                seq_succ_score /= valid_frames
                seq_norm_prec_score /= valid_frames
                seq_gen_succ_rob_score /= valid_frames

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_score': seq_succ_score,
                    'normalized_precision_score': seq_norm_prec_score,
                    'generalized_success_robustness_score': seq_gen_succ_rob_score}})

                overall_succ_score += (seq_succ_score * seq_length)
                overall_norm_prec_score += (seq_norm_prec_score * seq_length)
                overall_gen_succ_rob_score += (seq_gen_succ_rob_score * seq_length)

                overall_seq_length += seq_length

            overall_succ_score /= overall_seq_length
            overall_norm_prec_score /= overall_seq_length
            overall_gen_succ_rob_score /= overall_seq_length

            # store overall performance
            performance[name]['overall'].update({
                'success_score': overall_succ_score,
                'normalized_precision_score': overall_norm_prec_score,
                'generalized_success_robustness_score': overall_gen_succ_rob_score})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        return performance

    def _record(self, record_file, boxes, times=None):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        if times is not None:
            # record running times
            time_dir = os.path.join(record_dir, 'times')
            if not os.path.isdir(time_dir):
                os.makedirs(time_dir)
            time_file = os.path.join(time_dir, os.path.basename(
                record_file).replace('.txt', '_time.txt'))
            np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, boxes, anno):
        ious = []
        norm_center_errors = [] 

        for box, a in zip(boxes, anno):
            if a[0] < 0 and a[1] < 0 and a[2] < 0 and a[3] < 0:
                continue
            else:
                ious.append(rect_iou(np.array([box]), np.array([a]))[0])
                norm_center_errors.append(normalized_center_error(np.array([box]), np.array([a]))[0])

        ious = np.array(ious)
        norm_center_errors = np.array(norm_center_errors)
        
        return ious, norm_center_errors

    def _calc_curves(self, ious, norm_center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_nce = np.less_equal(norm_center_errors, thr_nce)

        succ_curve = np.mean(bin_iou, axis=0)
        norm_prec_curve = np.mean(bin_nce, axis=0)

        return succ_curve, norm_prec_curve

    def _calc_curves_robustness(self, ious):
        seq_length = ious.shape[0]

        thr_iou = np.linspace(0, 0.5, self.nbins_gsr)

        gen_succ_rob_curve = np.zeros(thr_iou.shape[0])
        for i, th in enumerate(thr_iou):
            broken = False
            for j, iou in enumerate(ious):
                if iou <= th:
                    gen_succ_rob_curve[i] = float(j) / seq_length
                    broken = True
                    break
            if not broken:
                gen_succ_rob_curve[i] = 1.0

        return gen_succ_rob_curve

    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance-ope.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.png')
        norm_prec_file = os.path.join(report_dir, 'normalized_precision_plots.png')
        gen_succ_rob_file = os.path.join(report_dir, 'generalized_success_robustness_plots.png')
        
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots of OPE')
        ax.grid(True)
        fig.tight_layout()
        
        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by normalized precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['normalized_precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_nce.max()), ylim=(0, 1),
               title='Normalized precision plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)


        # sort trackers by generalized success robustness score
        tracker_names = list(performance.keys())
        prec = [t[key]['generalized_success_robustness_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot generalized success robustness curves
        thr_succ_rob = np.linspace(0, 0.5, self.nbins_gsr)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_succ_rob,
                            performance[name][key]['generalized_success_robustness_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['generalized_success_robustness_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Normalized extent',
               xlim=(0, thr_succ_rob.max()), ylim=(0, 1),
               title='Generalized success robustness plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        print('Saving generalized robustness plots to', gen_succ_rob_file)
        fig.savefig(gen_succ_rob_file, dpi=300)