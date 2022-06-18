# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:10:27 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import os
import argparse
import pickle
import time

from experiments.Experiment import Experiment
from common.Config import Config
from common.Utils import Utils
from common.PushoverManager import PushoverManager


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Short-term timeseries prediction')
    parser.add_argument("-i --input-path", dest="input_path",
                        default="preprocessed-data", action="store",
                        help="Set input path.")
    parser.add_argument("-o --output-path", dest="output_path",
                        default="results", action="store",
                        help="Set output path.")
    parser.add_argument("-n --dataset-names", default=None, nargs='+',
                        dest="dataset_names", action="store", type=str,
                        help='Set the dataset names. \
                            (Default: %(default)s)')
    parser.add_argument("-m --model-names", default=None, nargs='+',
                        dest="model_names", action="store", type=str,
                        help='Set the model names. \
                            (Default: %(default)s)')
    parser.add_argument("-c --config-path", dest="config_path",
                        action="store",
                        default="conf",
                        help="Set config path.")
    parser.add_argument("-r --mc-runs", dest="mc_runs",
                        type=int,
                        action="store",
                        default=15,
                        help="Set MC runs number.")
    parser.add_argument("-e --error-metrics",
                        default=['maape', 'smape', 'mase'],
                        nargs='+',
                        dest="error_metrics", action="store",
                        type=str,
                        help='Set the error metrics. \
                            (Default: %(default)s)')
    parser.add_argument("--cv-error-metric",
                        dest="cv_error_metric",
                        action="store",
                        default="smape",
                        help="Set CV error metric.")
    parser.add_argument("-s --summary-stats",
                        default=['mean', 'median', 'std'],
                        nargs='+',
                        choices=['mean', 'median', 'std'],
                        dest="summary_stats", action="store", type=str,
                        help='Set the summary statistics. \
                            (Default: %(default)s)')
    parser.add_argument("--enable-gpu", dest="enable_gpu",
                        action='store_true',
                        help="Enable GPU for NN training.")
    parser.set_defaults(enable_gpu=False)
    return parser


def main(args=None):
    
    if args is None:
       args = argument_parser().parse_args()
           
    conf = Config(args.config_path)
    if not args.dataset_names:
        args.dataset_names = list(conf.get_dataset_config().keys())
    if not args.model_names:
        args.model_names = list(conf.get_model_config().keys())
    
    if args.cv_error_metric not in args.error_metrics:
        args.error_metrics.append(args.cv_error_metric)
    
    notifier = PushoverManager(
        conf.get_general_config()['notifications']['user_key'],
        conf.get_general_config()['notifications']['app_key'])
    
    total_exp_results = list()
    
    try:
        os.mkdir(args.output_path)
    except:
        pass
    
    for dataset in args.dataset_names:
        for model in args.model_names:
            
            try:
                experiment = Experiment(dataset,
                                        model,
                                        args.mc_runs,
                                        args.error_metrics,
                                        args.cv_error_metric,
                                        args.summary_stats,
                                        conf,
                                        datasets_path=args.input_path,
                                        enable_gpu=args.enable_gpu)

                experiment.run()
                
                res = {
                    'dataset': dataset,
                    'model': model,
                    'mcruns': args.mc_runs,
                    'exp_results': experiment.get_results()
                }
                
                # pickle & save experiment results
                with open(
                        os.path.join(args.output_path,
                            "{}-{}-{}".format(dataset,
                                              model,
                                              int(time.time()))), 'wb') as f:
                    pickle.dump(res, f)

                
                total_exp_results.append(res)
                
                # notify on experiment end
                if conf.get_general_config()['notifications']['enabled']:
                    notifier.notify_experiment_end(
                        dataset,
                        model,
                        int(experiment.get_duration()))
            except Exception as e:
                print('Exception: {}'.format(e))
                # notify on experiment error
                if conf.get_general_config()['notifications']['enabled']:
                    try:
                        notifier.notify_experiment_error(dataset,
                                                         model,
                                                         str(e))
                    except:
                        pass
    
    Utils.print_summary_statistics(total_exp_results,
                                   args.summary_stats)

    
if __name__ == "__main__":
    main()   
    