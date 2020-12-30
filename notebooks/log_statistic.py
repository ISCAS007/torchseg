# coding: utf-8
import os
import argparse

from torchseg.utils.summary_to_jupyter import dump

def dump_tasks(notes,
    delete_nan=False,
    dump_group=True,
    dump_dir=True,
    tags=['train/fmeasure','val/fmeasure'],
    descripe=['note','epoch'],
    rootpath=os.path.expanduser('~/tmp/logs/motion'),
    invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']):

    dump(tags=tags,
         rootpath=rootpath,
         notes=notes,
         descripe=descripe,
         invalid_param_list=invalid_param_list,
         delete_nan=delete_nan,
         dump_group=dump_group,
         dump_dir=dump_dir)

def dump_recent(
    tags=['train/fmeasure','val/fmeasure'],
    rootpath=os.path.expanduser('~/tmp/logs/motion'),
    notes=['today','recent'],
    dump_dir=True,
    recent_log_number=100,
    descripe=['note','epoch'],
    note_gtags=None):
    dump(tags=tags,
         rootpath=rootpath,
         notes=notes,
         note_gtags=note_gtags,
         sort_tags=[tags[1]],
         descripe=descripe,
         delete_nan=False,
         dump_group=False,
         dump_dir=dump_dir,
         recent_log_number=recent_log_number,)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--delete_nan',
                action='store_true')
    parser.add_argument('--dump_group',
                 action='store_true')
    parser.add_argument('--notes',
                nargs='*')
    parser.add_argument('--ignore_params',
                nargs='*',
                default=[])
    parser.add_argument('--recent_log_number',
                type=int,
                default=100)
    parser.add_argument('--dump_dir',
                action='store_true')
    parser.add_argument('--app',
                default='dump_recent',
                choices=['dump_tasks','dump_recent'])

    parser.add_argument('--tags',
                       choices=['f','iou','fps'],
                       default='f')

    args=parser.parse_args()

    if args.tags=='f':
        tags=['train/fmeasure','val/fmeasure']
        rootpath=os.path.expanduser('~/tmp/logs/motion')
        descripe=['note','epoch']
        invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']
        
    elif args.tags=='iou':
        tags=['train/iou','val/iou']
        rootpath=os.path.expanduser('~/tmp/logs/pytorch')
        descripe=['note','n_epoch']
        invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow','flow_backbone',
                    'ignore_outOfRoi','exception_value',
                    'cityscapes_split','txt_path',
                    'decode_main_layer','gpu','ngpus_per_node',
                    'rank']
        
    elif args.tags=='fps':
        tags=['train/fmeasure','val/fmeasure','train/fps','val/fps','total_param','train_param']
        rootpath=os.path.expanduser('~/tmp/logs/motion')
        descripe=['note','epoch']
        invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']
        

    if args.app=='dump_recent':
        recent_log_number=args.recent_log_number
        dump_recent(dump_dir=args.dump_dir,
                    rootpath=rootpath,
                    tags=tags,
                    recent_log_number=recent_log_number,
                    descripe=descripe)
    else:
        invalid_param_list+=args.ignore_params
        print('notes={},delete_nan={},dump_group={}'.format(args.notes,args.delete_nan,args.dump_group))
        dump_tasks(notes=args.notes,
            delete_nan=args.delete_nan,
            dump_group=args.dump_group,
            dump_dir=args.dump_dir,
            invalid_param_list=invalid_param_list,
            descripe=descripe,
            tags=tags,
            rootpath=rootpath)
