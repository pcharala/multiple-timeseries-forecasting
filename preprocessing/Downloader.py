# -*- coding: utf-8 -*-
"""
Created on Sat May 14 08:58:06 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import os
import sys
import shutil
import requests
import zipfile
from tqdm import tqdm
from importlib import import_module
from common.Config import Config

class Downloader():
    """
    A class that represents a downloader.
    
    Attributes
    ----------
    download_path: str
        the path to download original datasets.
    processed_path: str
        the path to sace pre-processed datasets
    dataset_names: list[str]
        the names of datasets to download/pre-process.
    config_path: str
        the path of the configuration files.
    delete_original: bool
        if True, delete original datasets after pre-processing.
    """
    
    def __init__(self,
                 download_path,
                 processed_path,
                 dataset_names,
                 config_path,
                 delete_original):
        
        self._download_path = download_path
        self._processed_path = processed_path
        self._conf = Config(config_path).get_dataset_config()
        self._dataset_names = \
            dataset_names if dataset_names else list(
                self._conf.keys())
        self._delete_original = delete_original
        
    
    @staticmethod
    def _preprocess_dataset(dataset_name,
                            input_filename,
                            output_filename):
        """
        Preprocess dataset

        """
        print('Pre-processing dataset: {} ...'.format(dataset_name), end=" ")
        module = import_module('preprocessing.preproc{}'
                               .format(dataset_name))
        module.preprocess(input_filename,
                          output_filename)
        print('OK')


    def download_and_preproc(self):
        """
        Download and pre-process the datasets.

        Returns
        -------
        None.

        """
        
        # download and unzip files
        try:
            os.mkdir(self._download_path)
        except:
            pass
        
        try:
            os.mkdir(self._processed_path)
        except:
            pass
        
        for dn in self._dataset_names:
            try:
                ds = self._conf[dn]
            except KeyError:
                print('Invalid dataset name: {}'.format(dn))
                continue
            
            print('---------------------------------------------------')
            print('Downloading files for dataset: {}'.format(dn))
            ds_subdir = os.path.join(self._download_path, dn)
            try:
                os.mkdir(ds_subdir)
            except:
                pass
            
            
            for url in ds['urls']:
                filesize = int(requests.head(url).headers["Content-Length"])
                filename = os.path.basename(url)
                with requests.get(url, stream=True) as r, \
                    open(os.path.join(ds_subdir, filename), "wb") as f,\
                        tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        total=filesize,
                        file=sys.stdout,
                        desc=filename
                        ) as progress:
                            for chunk in r.iter_content(chunk_size=1024):
                                datasize = f.write(chunk)
                                progress.update(datasize)
                                
                # if .zip, unzip file and delete original
                if os.path.splitext(filename)[-1] == '.zip':
                    print('Extracting {} ...'.format(filename), end=" ")
                    with zipfile.ZipFile(os.path.join(ds_subdir, filename)) as zf:
                        try: 
                            zf.extractall(ds_subdir)
                            # remove zip fle
                            print('OK')
                        except zipfile.error:
                            print('Failed')
                            continue
                    os.remove(os.path.join(ds_subdir, filename))
                                    
            # pre-process dataset
            try:
                os.mkdir(os.path.join(self._processed_path,
                                      dn))
            except:
                pass
            
            input_filenames = [os.path.join(ds_subdir, x)
                               for x in ds['input_filename']]
            Downloader._preprocess_dataset(dn,
                                           input_filenames,
                                           os.path.join(
                                               self._processed_path,
                                               dn,
                                               ds['output_filename']))
            if self._delete_original:
                shutil.rmtree(ds_subdir, ignore_errors=True)
            
        