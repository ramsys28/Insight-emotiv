#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:44:21 2023
Syntax: 
preproc('edf_flie.edf', lowEdge, highEdge, notch, IC_toRemove)
Inputs: 
'edf_flie.edf' -> edf file extracted by Insights headset
lowEdge -> Frequency of low cut filter
highEdge -> Frequency of high cut filter
notch -> Frequency of notch filter (50 or 60)
IC_toRemove -> Artifactual independent components to be removed*

Outputs:
A folder called results that contains....

*ATTENTION: When you first run the preproc.py use 0 for IC_toRemove in order 
not to remove any component from IC. Then Inspect ica.jpg in order to see which
component to remove and then run again preproc.py using as IC_toRemove the number of 
the artifactual component (starting from 1)

Usage by CMD:
>python preproc.py demo.edf 1 40 50 0
@author: kladosm
"""

import pandas as pd
import os
import mne
from mne.preprocessing import ICA
import sys
edf_file = sys.argv[1]
low = int(sys.argv[2])
high = int(sys.argv[3])
notch = int(sys.argv[4])
ic = int(sys.argv[5])



def import_insight_data(edf_file):
    
    data =  mne.io.read_raw_edf(edf_file)
    data.drop_channels(['TIME_STAMP_s',
     'TIME_STAMP_ms',
     'OR_TIME_STAMP_s',
     'OR_TIME_STAMP_ms',
     'COUNTER',
     'INTERPOLATED', 'RAW_CQ',
     'BATTERY',
     'BATTERY_PERCENT',
     'MarkerIndex',
     'MarkerType',
     'MarkerValueInt',
     'MARKER_HARDWARE',
     'CQ_AF3',
     'CQ_T7',
     'CQ_Pz',
     'CQ_T8',
     'CQ_AF4',
     'CQ_Overall',
     'EQ_SampleRateQua',
     'EQ_OVERALL',
     'EQ_AF3',
     'EQ_T7',
     'EQ_Pz',
     'EQ_T8',
     'EQ_AF4'])
    return data


def mneData_tocsv(data, filename):

    aaa = {data.ch_names[0]: data.get_data()[0],
           data.ch_names[1]: data.get_data()[1],
           data.ch_names[2]: data.get_data()[2],
           data.ch_names[3]: data.get_data()[3],
           data.ch_names[4]: data.get_data()[4],
          }
    df = pd.DataFrame.from_dict(aaa)
    if not os.path.isdir('results'):
        os.makedirs("results")
    df.to_csv(filename)
    
    
def main(edf_file, low, high, notch, ic):
   
    
    data = import_insight_data(edf_file)
    info = data.info
    mneData_tocsv(data, 'results/unfilteredEEG.csv')
    fig = data.plot(scalings = 0.0001, show_scrollbars=False)
    fig.savefig('results/unfiltered.jpg')
    data = data.crop(tmax=17)
    data_f = mne.filter.filter_data(data.get_data(), 128, low, high)
    data_f = mne.filter.notch_filter(data_f, 128, notch)
    data = mne.io.RawArray(data_f, info)
    fig = data.plot(scalings = 0.0001, show_scrollbars=False)
    fig.savefig('results/filtered.jpg')
    mneData_tocsv(data, 'results/filteredEEG.csv')
    mne.export.export_raw('results/filteredEEG.edf',data)
    
    ica = ICA(n_components=5, method='infomax')
    ica.fit(data)

    fig = ica.plot_sources(data, show_scrollbars=False)
    fig.savefig('results/ica.jpg')
    
    
    
    
    if ic>1:
        ica.exclude = [ic-1]
        ica.apply(data.load_data())
        mneData_tocsv(data, 'results/cleanEEG.csv')
        mne.export.export_raw('results/cleanedEEG.edf',data)
        fig = data.plot(scalings = 0.0001, show_scrollbars=False)
        fig.savefig('results/cleaned.jpg')

    montage = mne.channels.read_dig_fif('Insight_mont.fif')
    montage.ch_names[0] = 'AF3'
    montage.ch_names[1] = 'T7'
    montage.ch_names[2] = 'Pz'
    montage.ch_names[3] = 'AF4'
    montage.ch_names[4] = 'T8'
    data.set_montage(montage)
    fig = montage.plot()
    fig.savefig('results/montage.jpg')
    
main(edf_file, low, high, notch, ic)   
    
    
    
    
    
    
    
    
