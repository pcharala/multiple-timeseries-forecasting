# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:35:55 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import numpy as np
import time
import copy
import pickle

from .DatasetLoader import DatasetLoader

from .DataPreparation import DataPreparation
from .CrossValidator import CrossValidator
from .Trainer import Trainer

import tensorflow as tf
import os

class Experiment(object):
    """
    A class that represents an experiment.
    
    Attributes
    ----------
    dataset_name: str
        the name of the dataset.
    model_name: str
        the name of the model.
    mcruns: int
        the number of Monte-Carlo runs for current experiment.
    error_metrics: list[str]
        a list with error metrics to be computed.
    cv_decision_method: str
        the name of error metric used during cross-validation.
        Alternatively, 'majority_voting' for choosing best hyperparams
        combination based on majority voting on all error_metrics.
    conf_path: Config
        the configuration object.
    summary_stats: list[str]
        the summary stats of error metrics to calculate.
    datasets_path: str
        the parent path of datasets.
    enable_gpu: bool
        if True, enable GPU (if supported by processing backend,
                             e.g. TensorFlow).
    save_intermediate_res: bool
        if True, save results per Monte-Carlo run.
    intermediate_res_path: str
        Intermediate results path.
    """
    
    def __init__(self,
                 dataset_name,
                 model_name,
                 mcruns,
                 error_metrics,
                 cv_decision_method,
                 summary_stats,
                 conf,
                 datasets_path='data',
                 enable_gpu=True,
                 save_intermediate_res=False,
                 intermediate_res_path='temp-res'):
        
        self._dataset_name = dataset_name
        self._model = model_name
        self._mcruns = mcruns
        self._save_intermediate_res = save_intermediate_res
        self._intermediate_res_path = intermediate_res_path
        self._conf = conf.get_config(dataset_name,
                                     model_name,
                                     error_metrics,
                                     cv_decision_method,
                                     summary_stats)
        self._configure_accelerators(enable_gpu, self._conf['gpu_memory'])
        self._original_ts = self._load_and_preprocess_dataset(datasets_path)
        self._exp_results = list()
        self._duration = None
        self._finished = False

    
    def _configure_accelerators(self, enable_gpu=True, gpu_memory=5000):
        """
        Configure GPU accelerators.

        """
        if enable_gpu is not True:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_memory)])
                    logical_gpus = tf.config.experimental.list_logical_devices(
                        'GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus),
                          "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs
                    # have been initialized
                    print(e)
            else:
                print('TF: No GPUs were found. Using CPU instead.')


    
    def _load_and_preprocess_dataset(self, dataset_parent):
        """
        Load and preprocess dataset

        """
        
        dl = DatasetLoader(self._dataset_name,
                           os.path.join(dataset_parent,
                                        self._dataset_name,
                                        self._conf['dataset'][
                                            'output_filename'])
                           )
        
        
        return dl.dataset_load_and_preprocess()
        
        
    def run(self):
        """
        Run the experiment.

        Returns
        -------
        None.

        """
                
        # starting time
        st = time.time()
        
        # Set random generator seed for reproducibility reasons. Instead of
        # setting global rng seed, it is better to instantiate a separte rng.
        rng = np.random.RandomState(self._conf['random_seed'])
        
        for exp_window in self._conf['dataset']['exp_window']:
            
            print('---------------------------------------------------')
            print('EXP. WINDOW size: {}'.format(exp_window))
            
            # get max starting idx
            max_start_idx = self._original_ts.shape[1] - exp_window
            starting_idxs = rng.permutation(max_start_idx+1)[:self._mcruns]
            #print(starting_idxs[5:])
            #continue
            
            #################################################
            mcrun_results = {
                "exp_window": exp_window,
                "starting_idxs": starting_idxs,
                "metrics": {key: list() 
                                for key in self._conf['metrics']}
                }
            #################################################
            
            for l_idx, st_idx in enumerate(starting_idxs):
                                                
                print('MC RUN {}/{}'.format(l_idx+1, self._mcruns))
                
                sub_ts = self._original_ts[:, st_idx:st_idx+exp_window]
                        
                # Utils.print_ts_info(sub_ts)
                ###############################################################
                dp = DataPreparation(sub_ts,
                                     len_outsample=self._conf['len_outsample'],
                                     perc_valid=self._conf['perc_valid'],
                                     fh=self._conf['fh'],
                                     window_type=self._conf['window_type'],
                                     scaler=self._conf['scaler'],
                                     verbose=0)
                
                ###############################################################
                cvtor = CrossValidator(dp,
                                       copy.deepcopy(self._conf['model']),
                                       error_metrics=self._conf['metrics'],
                                       decision_method=
                                           self._conf['decision_method'],
                                       search_type=self._conf['search_type'],
                                       ncpus=self._conf['ncpus'])
                
                cvtor.run()
                best_hyperparams = cvtor.get_best_hyperparameters()
                ###############################################################
                # train on insample & predict on outsample
                MODEL = copy.deepcopy(self._conf['model'])
                MODEL['params'] = best_hyperparams
                
                mod = Trainer(dp, MODEL, error_metrics=self._conf['metrics'])
                res_dict = mod.train_and_predict()
                ###############################################################
                
                ############################################
                for metric in mcrun_results['metrics']:
                    mcrun_results['metrics'][metric].append(
                        res_dict['metrics'][metric])
                    
                ############################################
                
                # pickle & save experimental results for (exp_window, mcrun)
                if self._save_intermediate_res:
                    
                    if not os.path.exists(self._intermediate_res_path):
                        os.makedirs(self._intermediate_res_path)
                    
                    with open(
                            os.path.join(
                                self._intermediate_res_path,
                                "{}-{}-{}-w{}-r{}".format(
                                    int(time.time()),
                                    self._dataset_name,
                                    self._conf['model']['name'],
                                    exp_window,
                                    l_idx)
                                ), 'wb') as f:
                        pickle.dump(mcrun_results, f)
        
            self._exp_results.append(mcrun_results)
        
        self._duration = time.time() - st
        self._finished = True
        
        # Utils.print_summary_statistics(self._dataset_name,
        #                                self._model,
        #                                self._mcruns,
        #                                self._exp_results,
        #                                self._conf['summary_stats'])
    
    
    def get_results(self):
        """
        Get results of experiment.

        Raises
        ------
        Exception
            If experiment has not finished yet.

        Returns
        -------
        list[exp_res]
            List with results of experiment.

        """
        
        if not self._finished:
            raise Exception('Experiment has not finished yet!')
            
        return self._exp_results


    def get_duration(self):
        """
        Get duration of experiment.

        Raises
        ------
        Exception
            If experiment has not finished yet.

        Returns
        -------
        float
            Total duration of experiment in secs.

        """
        
        if not self._finished:
            raise Exception('Experiment has not finished yet!')
            
        return self._duration
            