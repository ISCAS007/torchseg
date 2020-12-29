# -*- coding: utf-8 -*-

import argparse
import sys

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--good')
    parser.add_argument('--sub_config')

    if len(sys.argv)==1:
        args=parser.parse_args(['--good','okay','--sub_config','-a 10 -b hello'])
    else:
        args=parser.parse_args()
        print(sys.argv)

    sub_parser=argparse.ArgumentParser()
    sub_parser.add_argument('-a',type=int)
    sub_parser.add_argument('-b',type=str)
    sub_args=sub_parser.parse_args(args.sub_config.split())
    print(args,sub_args)

