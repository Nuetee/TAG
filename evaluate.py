from data_configs import DATASETS
import argparse
import numpy as np
import json
import torch
from tqdm import tqdm
from vlm_localizer import localize
import os
from llm_prompting import select_proposal

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_llm', action='store_true', help='Enable use llm')
    parser.add_argument('--tckmeans', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def eval_without_llm(data, feature_path, stride, hyperparams, tckmeans):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)
            proposals = select_proposal(np.array(proposals))

            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval_with_llm(data, feature_path, stride, hyperparams, tckmeans):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())
    
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)
            
            if 'query_json' in ann['response'][i]:
                for j in range(len(ann['response'][i]['query_json'][0]['descriptions'])):
                    query_json = [{'descriptions': ann['response'][i]['query_json'][0]['descriptions'][j]}]
                    proposals += localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)

            proposals = select_proposal(np.array(proposals))
        
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))



def eval(data, feature_path, stride, hyperparams, use_llm, tckmeans, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)
            
            if use_llm:
                if 'query_json' in ann['response'][i]:
                    for j in range(len(ann['response'][i]['query_json'][0]['descriptions'])):
                        query_json = [{'descriptions': ann['response'][i]['query_json'][0]['descriptions'][j]}]
                        proposals += localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)

            proposals = select_proposal(np.array(proposals))

            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = proposals[0][0],  proposals[0][1]

            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))



if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)


    if args.llm_output and os.path.exists(args.llm_output):
        with open(args.llm_output) as f:
            data = json.load(f)
        if args.use_llm:
            eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.tckmeans)
        else:
            eval_without_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.tckmeans)
    else:
        with open(dataset['splits'][args.split]['annotation_file']) as f:
            data = json.load(f)
        eval(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.use_llm, args.tckmeans, dataset['splits'][args.split]['pad_sec'])