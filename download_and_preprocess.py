# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:03:16 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import argparse
from preprocessing.Downloader import Downloader


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Dataset downloading and preprocessing')
    parser.add_argument("-d --download-path", dest="download_path",
                        default="original-data", action="store",
                        help="Set download path.")
    parser.add_argument("-p --preproc-path", dest="preproc_path",
                        default="preprocessed-data", action="store",
                        help="Set preprocessing path.")
    parser.add_argument("-n --dataset-names", default=None, nargs='+',
                        dest="dataset_names", action="store", type=str,
                        help='Set the dataset names. \
                            (Default: %(default)s)')
    parser.add_argument("-c --config-path", dest="config_path",
                        action="store",
                        default="conf",
                        help="Set config path.")
    parser.add_argument('--del-original', dest="del_original",
                        action='store_true',
                        help="Delete original datasets.")
    parser.set_defaults(del_original=False)
    return parser


def main(downloader=Downloader, args=None):
    
    if args is None:
       args = argument_parser().parse_args()

    d = downloader(
            download_path=args.download_path,
            processed_path=args.preproc_path,
            dataset_names=args.dataset_names,
            config_path=args.config_path,
            delete_original=args.del_original
            )
    d.download_and_preproc()


if __name__ == "__main__":
    main()   
    