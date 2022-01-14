# This file is part of me-features-to-mol-ID-mapping.
#
#
# Copyright © 2021 Blue Brain Project/EPFL
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the APACHE-2 License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

import glob
import pandas as pd
import os
import h5py
from bluepyefe.extract import extract_efeatures
from scipy.interpolate import interp1d
import numpy
import numpy as np


def nwb_reader_BBP(in_data):

    filepath = in_data['filepath']
    r = h5py.File(filepath, 'r')

    data = []
    for sweep in list(r['acquisition'].keys()):
        protocol_name = str(
            r['acquisition'][sweep].attrs['stimulus_description'])
        if protocol_name == in_data['protocol_name']:
            trace_data = {}
            trace_data['voltage'] = numpy.array(
                r['acquisition'][sweep]['data'][()],
                dtype="float32")
            _ = '__'.join([sweep.split('__')[0] + 's', sweep.split('__')[1], sweep.split('__')[2]])
            trace_data['current'] = numpy.array(
                r['stimulus']['presentation'][_]['data'][()],
                dtype="float32")
            trace_data['dt'] = 1. / float(
                r['acquisition'][sweep]['starting_time'].attrs["rate"])
            data.append(trace_data)
            trace_data['i_unit'] = "pA"
            trace_data['t_unit'] = "s"
            trace_data['v_unit'] = "mV"

    return data

def nwb_reader_Gouw(in_data):
    filepath = in_data['filepath']
    r = h5py.File(filepath, 'r')
    data = []
    for sweep in list(r['acquisition']['timeseries'].keys()):
        protocol_name = str(r['acquisition']['timeseries'][sweep]['aibs_stimulus_name'][()])
        current_sweep = 'Experiment_{}'.format(sweep.replace('Sweep_', ''))
        if protocol_name == in_data['protocol_name']:
            voltage = numpy.array(
                    r['acquisition']['timeseries'][sweep]['data'][()],
                    dtype="float32"
            )
            current = numpy.array(
                    r['epochs'][current_sweep]['stimulus']['timeseries']['data'],
                    dtype="float32"
                )
            dt = 1. / float(
                    r['acquisition']['timeseries'][sweep]['starting_time'].attrs[
                    "rate"])
            new_dt = 0.0001
            old_time = [dt * i for i in range(len(voltage))]
            new_time = numpy.arange(0., old_time[-1], new_dt)
            f_voltage = interp1d(
                old_time,
                voltage,
                fill_value='extrapolate'
            )
            f_current = interp1d(
                old_time,
                current,
                fill_value='extrapolate'
            )
            trace_data = {
                'voltage': numpy.array(f_voltage(new_time), dtype="float32"),
                'current': numpy.array(f_current(new_time), dtype="float32"),
                'dt': new_dt
            }
            del current
            del voltage
            data.append(trace_data)
    r.close()
    return data

def prepare_BBP_config_file(target_amp_list, tol, cell_name_list,
                            path="../downloader/BBP/BBP_ephys_traces/*"):

    meta_data = {}

    for i, file in enumerate(glob.glob(path)):
        cell_name = file.split("/")[-1].split(".")[0]
        if cell_name in cell_name_list:
            _ = os.path.basename(file)
            try:
                r = h5py.File(file, 'r')

                meta_data[_] = {'IDRest':
                    [{"filepath": file,
                      "eCode": "IDRest",
                      "protocol_name": "IDRest"
                      }]
                                }

            except:
                print('Cell', _, 'problematic file')

        # if (len(meta_data)>9):
        #     break

    target_list = []
    for efeat in interesting_efeatures:
        for amp in target_amp_list:
            target_list.append({
                "efeature": efeat,
                "protocol": "IDRest",
                "amplitude": amp,
                "tolerance": tol,
                # "efel_settings": {
                #     'stim_start': 200.,
                #     'stim_end': 500.,
                #     'Threshold': -10.
                # }
            })

    config = {'options': {'logging': True,
                          'protocols_threshold': ['IDRest']},
              'targets' : target_list,
              'meta_data': meta_data
             }
    return config

def prepare_Gouw_config_file(target_amp_list, tol, path="../downloader/Gouwens_2019/ephys_traces/*"):
    
    meta_data = {}
    for i, file in enumerate(glob.glob(path)):
        _ = os.path.basename(file)
        meta_data[_] = {'step':[{"filepath": file,
                              "i_unit": "A",
                              "v_unit": "V",
                              "t_unit": "s",
                              "ljp": 14.,
                              "protocol_name": 'Long Square',
                              "ton": 1000.,
                              "toff": 2000.}]}
        # if (len(meta_data)>9):
        #     break

    target_list = []
    for efeat in interesting_efeatures:
        for amp in target_amp_list:
            target_list.append({
                "efeature": efeat,
                "protocol": "step",
                "amplitude": amp,
                "tolerance": tol,
                # "efel_settings": {
                #     'stim_start': 200.,
                #     'stim_end': 500.,
                #     'Threshold': -10.
                # }
            })

    config = {'options': {'format': 'nwb',
                          'logging': True,
                          'protocols_threshold': ['Long Square']},
              'targets': target_list,
              'meta_data': meta_data
             }
    return config

def make_dict_BBP(extractor, target_list, tolerance):
    step_dict = {}

    for cell in extractor.cells:
        cell_id = cell.name.split('.')[0]
        step_dict['Cell_BBP_' + str(cell_id)] = {}

        for target in target_list:
            trace_list = []
            for trace in cell.traces:
                if (target - tolerance) < trace.ecode.amp/cell.amp_threshold < (target + tolerance):
                    trace_list.append(trace.efeatures)
            print('BBP', str(target) + ' : ' + str(len(trace_list)))
            if len(trace_list) > 1:
                tmp_dict = {}
                for feature in trace_list[0].keys():
                    #feature_values = [dict_[feature] for dict_ in trace_list]
                    tmp_dict[feature] = {'mean' : np.mean([dict_[feature] for dict_ in trace_list]),
                                         'sd' : np.std([dict_[feature] for dict_ in trace_list]),
                                         'N' : len(trace_list)}

                step_dict['Cell_BBP_' + str(cell_id)][str(target)] = tmp_dict

            elif len(trace_list) == 1:
                step_dict['Cell_BBP_' + str(cell_id)][str(target)] = trace_list[0]

            else:
                step_dict['Cell_BBP_' + str(cell_id)][str(target)] = ['No_data']


        print('________________')
    return step_dict

def make_dict_Gouw(extractor, target_list, tolerance):
    step_dict = {}

    for cell in extractor.cells:

        step_dict['Cell_AIBS_' + str(cell.name.split('_')[-1][5:])] = {}

        for target in target_list:
            trace_list = []
            for trace in cell.traces:
                if (target - tolerance) < trace.ecode.amp/cell.amp_threshold < (target + tolerance):
                    trace_list.append(trace.efeatures)
            print('Gouwens', str(target) + ' : ' + str(len(trace_list)))
            if len(trace_list) > 1:
                tmp_dict = {}
                for feature in trace_list[0].keys():
                    #feature_values = [dict_[feature] for dict_ in trace_list]
                    tmp_dict[feature] = {'mean' : np.mean([dict_[feature] for dict_ in trace_list]),
                                         'sd' : np.std([dict_[feature] for dict_ in trace_list]),
                                         'N' : len(trace_list)}

                step_dict['Cell_AIBS_' + str(cell.name.split('_')[-1][5:])][str(target)] = tmp_dict

            elif len(trace_list) == 1:
                step_dict['Cell_AIBS_' + str(cell.name.split('_')[-1][5:])][str(target)] = trace_list[0]

            else:
                step_dict['Cell_AIBS_' + str(cell.name.split('_')[-1][5:])][str(target)] = ['No_data']


        print('________________')
    return step_dict



if __name__=="__main__":
    
    import logging
    logger = logging.getLogger()
    logging.basicConfig(
            level=logging.DEBUG,
            handlers=[logging.StreamHandler()]
        )
    
    ephys_metadata_BBP = pd.read_csv('../downloader/BBP/metadata_electrophysiology.csv',
                                     index_col=0)

    BBP_etype_list = ephys_metadata_BBP["annotation.hasBody.label"]
    BBP_cell_name_list = ephys_metadata_BBP["derivation.entity.name"]


    interesting_efeatures = pd.read_csv('./BPEfe2_features_AP_fqcy.csv',
                                        index_col=0)['features_name'].tolist()

    Target_amplitudes = [80, 100, 125, 150, 175, 200, 225, 250, 275, 300]

    ### TO DO: extract BBP e feature per BBP etypes
    config_dict = {}
    for BBP_etype in BBP_etype_list.unique():
        msk_ = np.asarray([etype == BBP_etype for etype in BBP_etype_list])
        # print(BBP_etype)
        # print(BBP_cell_name_list[msk_].tolist())
        # print("_____________")
        # BBP_etype_dict[BBP_etype] = BBP_cell_name_list[msk_]
        config_dict[BBP_etype] = prepare_BBP_config_file(target_amp_list=Target_amplitudes,
                                                         tol=20,
                                                         cell_name_list=BBP_cell_name_list[msk_].tolist())

    config_dict["Gouwens"] = prepare_Gouw_config_file(target_amp_list=Target_amplitudes,
                                                      tol=20)

    for config_ in config_dict.keys():

        if config_ in BBP_etype_list.unique():
            config_BBP = config_dict[config_]
            efeatures_BBP, protocol_definitions_BBP, current_BBP = extract_efeatures(
                output_directory = "./BBP_efeatures/" + config_ + "/",
                files_metadata=config_BBP['meta_data'],
                targets=config_BBP['targets'],
                threshold_nvalue_save=1,
                protocols_rheobase=['IDRest'],
                recording_reader=nwb_reader_BBP,
                map_function=map,
                write_files=True,
                plot=False,
                low_memory_mode=False,
                spike_threshold_rheobase=1,
                protocol_mode="mean",
                efel_settings=None,
                extract_per_cell=False
            )

        if config_ == "Gouwens":
            config_Gouw = config_dict["Gouwens"]
            efeatures_Gouwens, protocol_definitions_Gouwens, current_Gouwens = extract_efeatures(
                output_directory="./Gouwens_efeatures/",
                files_metadata=config_Gouw['meta_data'],
                targets=config_Gouw['targets'],
                threshold_nvalue_save=1,
                protocols_rheobase=['step'],
                recording_reader=nwb_reader_Gouw,
                map_function=map,
                write_files=True,
                plot=False,
                low_memory_mode=True,
                spike_threshold_rheobase=1,
                protocol_mode="mean",
                efel_settings=None,
                extract_per_cell=True
            )
