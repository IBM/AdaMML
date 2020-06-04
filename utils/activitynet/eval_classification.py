import json
import urllib.request, urllib.error, urllib.parse
import concurrent.futures

import numpy as np
import pandas as pd

API = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/challenge19/api.py'


def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib.request.Request(api_url)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def ap_compute_wrapper(activity, cidx, ground_truth, prediction):
    gt_idx = ground_truth['label'] == cidx
    pred_idx = prediction['label'] == cidx
    out = compute_average_precision_classification(ground_truth.loc[gt_idx].reset_index(drop=True),
                                                   prediction.loc[pred_idx].reset_index(drop=True))
    return out, cidx


class ANETclassification(object):
    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 subset='validation', verbose=False, top_k=3,
                 check_status=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.top_k = top_k
        self.ap = None
        self.hit_at_k = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        #self.output_file = open(prediction_filename[:-3] + "out", 'w')

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        activity_index, cidx = {}, 0
        video_lst, label_lst = [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                label_lst.append(activity_index[ann['label']])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        #with open(prediction_filename, 'r') as fobj:
        #    data = json.load(fobj)

        data = prediction_filename
        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        video_lst, label_lst, score_lst = [], [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros(len(list(self.activity_index.items())))
        with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(ap_compute_wrapper, activity, cidx, self.ground_truth, self.prediction) 
                                       for activity, cidx in self.activity_index.items()]
            for future in concurrent.futures.as_completed(futures):
                ap_v, cidx = future.result()
                ap[cidx] = ap_v                
        """
        for activity, cidx in self.activity_index.items():
            gt_idx = self.ground_truth['label'] == cidx
            pred_idx = self.prediction['label'] == cidx
            ap[cidx] = compute_average_precision_classification(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True))
        """
        return ap

    def evaluate(self, map_only):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        ap = self.wrapper_compute_average_precision()
        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet untrimmed video '
                   'classification task.')
            print('\tMean Average Precision: {}'.format(ap.mean()))
        #print('Mean Average Precision: {}'.format(ap.mean()), file=self.output_file)
        self.ap = ap
        
        if map_only:
            return ap.mean()
        
        hit_at_k, avg_hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        #avg_hit_at_k = compute_video_hit_at_k(
        #    self.ground_truth, self.prediction, top_k=self.top_k, avg=True)
        if self.verbose:
            print('\tHit@{}: {}'.format(self.top_k, hit_at_k))
            print('\tAvg Hit@{}: {}'.format(self.top_k, avg_hit_at_k))
        #print('Avg Hit@{}: {}'.format(self.top_k, avg_hit_at_k), file=self.output_file)
        self.hit_at_k = hit_at_k
        self.avg_hit_at_k = avg_hit_at_k

        return ap.mean(), self.avg_hit_at_k

################################################################################
# Metrics
################################################################################

def compute_average_precision_classification(ground_truth, prediction):
    """Compute average precision (classification task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matched as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'score']

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones(len(ground_truth)) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)


    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Assigning true positive to truly grount truth instances.
    for idx in range(len(prediction)):
        this_pred = prediction.loc[idx]
        gt_idx = ground_truth['video-id'] == this_pred['video-id']
        # Check if there is at least one ground truth in the video associated.
        if not gt_idx.any():
            fp[idx] = 1
            continue
        this_gt = ground_truth.loc[gt_idx].reset_index()
        if lock_gt[this_gt['index']] >= 0:
            fp[idx] = 1
        else:
            tp[idx] = 1
            lock_gt[this_gt['index']] = idx

    # Computing prec-rec
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / npos
    prec = tp / (tp + fp)
    return interpolated_prec_rec(prec, rec)


def hit_at_k_wrapper(vid, i, prediction, ground_truth, top_k):
    pred_idx = prediction['video-id'] == vid
    if not pred_idx.any():
        return None, None, i
    this_pred = prediction.loc[pred_idx].reset_index(drop=True)
    # Get top K predictions sorted by decreasing score.
    sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
    this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
    # Get labels and compare against ground truth.
    pred_label = this_pred['label'].tolist()
    gt_idx = ground_truth['video-id'] == vid
    gt_label = ground_truth.loc[gt_idx]['label'].tolist()
    out = np.mean([1 if this_label in pred_label else 0
                                   for this_label in gt_label])
    out2 = np.ceil(out)

    return out, out2, i


def compute_video_hit_at_k(ground_truth, prediction, top_k=3, avg=False):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)

    a_avg_hits_per_vid = np.zeros(video_ids.size)
    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(hit_at_k_wrapper, vid, i, prediction, ground_truth, top_k) 
                   for i, vid in enumerate(video_ids)]
        for future in concurrent.futures.as_completed(futures):
            out, out2, i = future.result()
            if out is not None:
                avg_hits_per_vid[i] = out
                a_avg_hits_per_vid[i] = out2

    """
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_vid[i] = np.mean([1 if this_label in pred_label else 0
                                       for this_label in gt_label])
        if not avg:
            avg_hits_per_vid[i] = np.ceil(avg_hits_per_vid[i])
    """
    return float(avg_hits_per_vid.mean()), float(a_avg_hits_per_vid.mean())


def compute_map(ground_truth_filename, prediction_filename,
                subset='validation', verbose=True, check_status=True, top_k=1, map_only=True):
    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=subset, verbose=verbose,
                                             check_status=check_status,
                                             top_k=top_k)
    return anet_classification.evaluate(map_only=map_only)
