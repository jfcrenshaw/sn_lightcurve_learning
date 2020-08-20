#!/usr/bin/env python3

import numpy as np
from astropy.table import Table
from classes import SkyObject
import pickle

data_loc = 'plasticc_data/'
metadata_files = ['plasticc_train_metadata.csv', 'plasticc_test_metadata.csv']
data_files = ['plasticc_train_lightcurves.csv', 'plasticc_test_set_batch1.csv']

source_dict = {90:'SNIa', 67:'SNIa-91bg', 52:'SNIax', 42:'SNII', 62:'SNIbc', 95:'SLSN-I'}
filter_dict = {0:'lsstu', 1:'lsstg', 2:'lsstr', 3:'lssti', 4:'lsstz', 5:'lssty'}

output_dict = {source:[] for source in source_dict.values()}

for mdfile, dfile in zip(metadata_files, data_files):

    print("Processing", dfile)

    metadata = Table.read(data_loc + mdfile)
    data = Table.read(data_loc + dfile)

    if mdfile == 'plasticc_test_metadata.csv':
        metadata = metadata[metadata['ddf_bool'] == 1]

    for key,source in source_dict.items():
        
        objs = metadata[metadata['true_target'] == key]['object_id','true_target','true_submodel',
                                                        'true_peakmjd','true_z','hostgal_photoz',
                                                        'hostgal_photoz_err','true_distmod']

        for row in objs:
            obj = SkyObject()
            obj.specz = row['true_z']
            obj.photoz = row['hostgal_photoz']
            obj.photoz_err = row['hostgal_photoz_err']
            obj.source = source if key != 42 else f"{source}-{row['true_submodel']}"
            obj.t0 = row['true_peakmjd']
            obj.distmod = row['true_distmod']

            photometry = data[data['object_id'] == row['object_id']]
            photometry['passband'] = [filter_dict[i] for i in photometry['passband']]
            photometry.rename_column('passband', 'filter')
            photometry['flux_err'] = np.abs(photometry['flux_err'])
            del photometry['object_id','detected_bool']

            norm = 1e-18
            photometry['flux'] *= 10**(2/5*obj.distmod) * norm
            photometry['flux_err'] *= 10**(2/5*obj.distmod) * norm

            obj.photometry = photometry
            
            output_dict[source].append(obj)
        

for source in source_dict.values():

    filename = f'plasticc_data/{source}_SkyObjects.pkl' 
    print(f"Saving {filename}")
    with open(filename, 'wb') as output:
        pickle.dump(output_dict[source], output)
