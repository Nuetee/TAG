DATASETS={
    'charades': {
        'feature_path': './datasets/Charades/',
        'stride': 20,
        'max_stride_factor': 0.5,
        'hyper_parameters': {
            'stride': 20,
            "gamma": 0.2,
            "cand_num": 12,
            "kmeans_k": 9,
            "window_radius": 7,
            "prior": 0.5,
            "temporal_window_size": 21,
            'is_clip': False,
            'is_blip': False,
            'is_blip2': True,
            'is_internVideo': False
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'charades_clip': {
        'feature_path': './datasets/Charades_CLIP/',
        'stride': 20,
        'max_stride_factor': 0.5,
        'hyper_parameters': {
            'stride': 20,
            "gamma": 0.2,
            "cand_num": 12,
            "kmeans_k": 9,
            "prior": 0.5,
            "temporal_window_size": 21,
            'is_clip': True,
            'is_blip': False,
            'is_blip2': False,
            'is_internVideo': False
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'charades_blip': {
        'feature_path': './datasets/Charades_BLIP/',
        'stride': 20,
        'max_stride_factor': 0.5,
        'hyper_parameters': {
            'stride': 20,
            "gamma": 0.2,
            "cand_num": 12,
            "kmeans_k": 9,
            "prior": 0.5,
            "temporal_window_size": 21,
            'is_clip': False,
            'is_blip': True,
            'is_blip2': False,
            'is_internVideo': False
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'charades_internVideo': {
        'feature_path': './datasets/Charades_InternVideo/',
        'stride': 20,
        'max_stride_factor': 0.5,
        'hyper_parameters': {
            'stride': 20,
            "gamma": 0.2,
            "cand_num": 12,
            "kmeans_k": 9,
            "prior": 0.5,
            "temporal_window_size": 21,
            'is_clip': False,
            'is_blip': False,
            'is_blip2': False,
            'is_internVideo': True
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'activitynet': {
        'feature_path': './datasets/ActivityNet/',
        'stride': 40,
        'max_stride_factor': 1,
        'hyper_parameters': {
            'stride': 40,
            "gamma": 0.8,
            "cand_num": 17,
            "kmeans_k": 9,
            "window_radius": 7,
            "prior": 1,
            "temporal_window_size": 21,
            'is_clip': False,
            'is_blip': False,
            'is_blip2': True,
            'is_internVideo': False
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 30.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 60.0,
            },
        }
    },
    'qvhighlight': {
        'feature_path': './datasets/QVHighlights_features/',
        'stride': 50,
        'max_stride_factor': 0.5,
        'hyper_parameters': {
            'stride': 40,
            "gamma": 0.8,
            "cand_num": 13,
            "kmeans_k": 25,
            "prior": 1,
            "temporal_window_size": 25,
            'is_clip': False,
            'is_blip': False,
            'is_blip2': True,
            'is_internVideo': False
        },
        'splits': {
            'default': {
                'annotation_file': 'dataset/qvhighlight/highlight_val_release_aug.jsonl',
                'pad_sec': 0.0,
            },
        }
    },
}