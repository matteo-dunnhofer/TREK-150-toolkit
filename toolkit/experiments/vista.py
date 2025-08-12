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
import tqdm
from pycocotools.mask import encode, decode

from ..datasets import VISTA
from ..utils.metrics import rect_iou, center_error, normalized_center_error, segm_iou, segm_iou_vec, normalized_center_error_segm
from ..utils.viz import show_frame
from ..utils.ioutils import compress_file

class ExperimentVISTA(object):
    r"""Experiment pipeline and evaluation toolkit for the VISTA dataset.

    Args:
        root_dir (string): Root directory of VISTA dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, split, mode='lt', anno_type='box', result_dir='./results', report_dir='./reports'):
        super(ExperimentVISTA, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.anno_type = anno_type
        self.dataset = VISTA(root_dir, split=split, mode=mode, anno_type=anno_type)
        self.result_dir = os.path.join(result_dir, 'VISTA')
        self.report_dir = os.path.join(report_dir, 'VISTA')
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_nce = 51
        self.nbins_gsr = 51


    def run_sope(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        self.dataset.just_first_anno = True

        # loop over the complete dataset
        for s, (img_files_fpv, anno_fpv, img_files_tpv, anno_tpv, frame_idxs) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('*** Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file_fpv = os.path.join(
                self.result_dir, tracker.name, f'sope-{self.split}-{self.mode}', seq_name, 'fpv.json')
            if os.path.exists(record_file_fpv):
                print('  Found FPV results, skipping FPV on', seq_name)
            else:
                # tracking loop
                preds_fpv, _ = tracker.track(
                    img_files_fpv, anno_fpv[0, :], visualize=visualize)
                assert len(preds_fpv) == len(img_files_fpv)

                # record results
                self._record(record_file_fpv, preds_fpv, frame_idxs)

            # skip if results exist
            record_file_tpv = os.path.join(
                self.result_dir, tracker.name, f'sope-{self.split}-{self.mode}', seq_name, 'tpv.json')
            if os.path.exists(record_file_tpv):
                print('  Found TPV results, skipping TPV on', seq_name)
            else:
                # tracking loop
                preds_tpv, _ = tracker.track(
                    img_files_tpv, anno_tpv[0, :], visualize=visualize)
                assert len(preds_tpv) == len(img_files_tpv)

                # record results
                self._record(record_file_tpv, preds_tpv, frame_idxs)
    
    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        self.dataset.just_first_anno = False

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-sope-{self.split}-{self.mode}.json')
        
        performance = {}
        for name in tracker_names:
            print('*** Evaluating', name)

            seq_num = len(self.dataset)
            aucs = np.zeros((seq_num, 2))
            npss = np.zeros((seq_num, 2))
            gsrs = np.zeros((seq_num, 2))
            weights = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, output in tqdm.tqdm(enumerate(self.dataset), total=len(self.dataset)):
                _, anno_fpv, _, anno_tpv, frame_idxs = output

                seq_name = self.dataset.seq_names[s]

                performance[name]['seq_wise'].update({seq_name: {} })

                for p, pov in enumerate(['fpv', 'tpv']):
                    if pov == 'fpv':
                        anno = anno_fpv
                    else:
                        anno = anno_tpv
                
                    record_file = os.path.join(self.result_dir, name, f'sope-{self.split}-{self.mode}', seq_name, f'{pov}.json')

                    with open(record_file, 'r') as f:
                        result_dict = json.load(f)

                    preds = []
                    valid_anno = []
                    #for i in range(anno.shape[0]):
                    for i, fi in enumerate(frame_idxs):
                        if not np.isnan(anno[i].sum()):
                            valid_anno.append(anno[i])
                            if self.anno_type == 'box':
                                preds.append(result_dict[fi])
                            else:
                                preds.append(decode(result_dict[fi]))
                    preds = np.array(preds)
                    valid_anno = np.array(valid_anno)

                    preds[0] = valid_anno[0, :]
                    assert len(preds) == len(valid_anno)

                    if self.anno_type == 'box':
                        ious, norm_center_errors = self._calc_metrics(preds, valid_anno)
                    else:
                        ious, norm_center_errors = self._calc_metrics_segm(preds, valid_anno)
                    succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                    gen_succ_rob_curve = self._calc_curves_robustness(ious)


                    aucs[s,p] = np.mean(succ_curve) * 100
                    npss[s,p] = np.mean(norm_prec_curve) * 100
                    gsrs[s,p] = np.mean(gen_succ_rob_curve) * 100
                    weights[s] = len(valid_anno)

                    # store sequence-wise performance
                    performance[name]['seq_wise'][seq_name].update({
                        pov : {
                            'intersection_over_union_auc': aucs[s,p],
                            'normalized_precision_score': npss[s,p],
                            'generalized_success_robustness_score': gsrs[s,p] 
                        }})

                performance[name]['seq_wise'][seq_name].update({
                        'difference': {
                            'delta_auc': aucs[s,0] - aucs[s,1],
                            'delta_nps': npss[s,0] - npss[s,1],
                            'delta_gsr': gsrs[s,0] - gsrs[s,1] 
                        }})
                
            avg_auc_fpv = (aucs[:,0] * weights).sum() / np.sum(weights)
            avg_nps_fpv = (npss[:,0] * weights).sum() / np.sum(weights)
            avg_gsr_fpv = (gsrs[:,0] * weights).sum() / np.sum(weights)

            avg_auc_tpv = (aucs[:,1] * weights).sum() / np.sum(weights)
            avg_nps_tpv = (npss[:,1] * weights).sum() / np.sum(weights)
            avg_gsr_tpv = (gsrs[:,1] * weights).sum() / np.sum(weights)

            delta_auc = avg_auc_fpv - avg_auc_tpv
            delta_nps = avg_nps_fpv - avg_nps_tpv
            delta_gsr = avg_gsr_fpv - avg_gsr_tpv
            
            # store overall performance
            performance[name]['overall'].update({
                'fpv': {
                    'avg_auc': avg_auc_fpv,
                    'avg_nps': avg_nps_fpv,
                    'avg_gsr': avg_gsr_fpv
                },
                'tpv': {
                    'avg_auc': avg_auc_tpv,
                    'avg_nps': avg_nps_tpv,
                    'avg_gsr': avg_gsr_tpv
                },
                'difference' : {
                    'delta_auc': delta_auc,
                    'delta_nps': delta_nps,
                    'delta_gsr': delta_gsr
                }
                })

            print('*** OK!')
            print(f'*** Performance available at {report_file}')    
            
        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        
        # plot performance difference plot
        self.performance_difference_plot(tracker_names)

        return performance


    def _record(self, record_file, preds, frame_idxs):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)

        if len(preds.shape) == 2:
            result_dict = {
                fi : pred.tolist() for fi, pred in zip(frame_idxs, preds)
            }
        else:
            result_dict = {}
            for fi, pred in zip(frame_idxs, preds):
                encoded_pred = encode(np.asfortranarray(pred.astype(np.uint8)))
                encoded_pred['counts'] = str(encoded_pred['counts'], "utf-8")
                result_dict[fi] = encoded_pred

        with open(record_file, 'w') as f:
            json.dump(result_dict, f, indent=4)

        print('  Results recorded at', record_file)


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

    def _calc_metrics_segm(self, segm, anno_segm):
 
        ious = segm_iou_vec(segm, anno_segm)
        norm_center_errors = normalized_center_error_segm(segm, anno_segm)
        
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

    def performance_difference_plot(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, f'performance-sope-{self.split}-{self.mode}.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        for metric in ['auc', 'nps', 'gsr']:
            pdp_file = os.path.join(report_dir, f'performance_difference_plot_{metric}-{self.split}-{self.mode}.png')

            key = 'overall'

            # markers
            markers = ['.', '+', '*']
            markers = [c + m for m in markers for c in [''] * 10]

            # sort trackers by success score
            tracker_names = list(performance.keys())
            succ = [t[key]['difference'][f'delta_{metric}'] for t in performance.values()]
            inds = np.argsort(succ)[::-1]
            tracker_names = [tracker_names[i] for i in inds]

            # plot scatter
            fig, ax = plt.subplots()
            ax.plot([0, 100], [0, 100], linestyle='--', color='black')
            dots = []
            legends = []
            for i, name in enumerate(tracker_names):
                dot, = ax.plot(performance[name][key]['tpv'][f'avg_{metric}'],
                                performance[name][key]['fpv'][f'avg_{metric}'],
                                markers[i % len(markers)], markersize=10)
                dots.append(dot)
                legends.append('%s: [%.1f, %.1f, %.1f]' % (name, performance[name][key]['fpv'][f'avg_{metric}'], performance[name][key]['tpv'][f'avg_{metric}'], performance[name][key]['difference'][f'delta_{metric}']))
            matplotlib.rcParams.update({'font.size': 7.4})
            legend = ax.legend(dots, legends, loc='center left',
                            bbox_to_anchor=(1, 0.5))

            matplotlib.rcParams.update({'font.size': 9})
            ax.set(xlabel=f'tpv-{metric}',
                ylabel=f'fpv-{metric}',
                xlim=(0, 100), ylim=(0, 100),
                title='Performance Difference Plot of SOPE')
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            
            print('*** Saving performance difference plot to', pdp_file)
            fig.savefig(pdp_file,
                        bbox_extra_artists=(legend,),
                        bbox_inches='tight',
                        dpi=300)

    
