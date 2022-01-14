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

from me_types_mapper.feature_selection_tools import *

def compute_labels():
    dict_Gouwens_ME = {
        'ME_Inh_1': 'BTC Vip',
        'ME_Inh_2': 'BTC Vip',
        'ME_Inh_3': 'BTC Vip',
        'ME_Inh_4': 'BTC Vip',
        'ME_Inh_5': 'BTC Vip',
        'ME_Inh_6': 'BC Pvalb',
        'ME_Inh_7': 'BC Pvalb',
        'ME_Inh_8': 'BC Pvalb',
        'ME_Inh_9': 'BC Pvalb',
        'ME_Inh_10': 'BC Pvalb',
        'ME_Inh_11': 'BC Pvalb',
        'ME_Inh_12': 'BC Pvalb',
        'ME_Inh_13': 'BC Pvalb',
        'ME_Inh_14': 'BC Pvalb',
        'ME_Inh_15': 'MC Sst',
        'ME_Inh_16': 'BC Pvalb',
        'ME_Inh_17': 'NGC Lamp5',
        'ME_Inh_18': 'NGC Lamp5',
        'ME_Inh_19': 'NGC Lamp5',
        'ME_Inh_20': 'NGC Lamp5',
        'ME_Inh_21': 'ChC Pvalb',
        'ME_Inh_22': 'nonMC Sst',
        'ME_Inh_23': 'nonMC Sst',
        'ME_Inh_24': 'MC Sst',
        'ME_Inh_25': 'MC Sst',
        'ME_Inh_26': 'MC&nonMC Sst'

    }

    dict_common_names = {
        'MC': 'MC',
        'BC': 'BC',
        'NGC': 'NGC',
        'NGC-DA': 'NGC',
        'NGC-SA': 'NGC',
        'DAC': 'NGC',
        'HAC': 'NGC',
        'DLAC': 'NGC',
        'SLAC': 'NGC',
        'ChC': 'ChC',
        'nonMC': 'BTC',  # 'BTC|BC|NGC|ChC',
        'MC&nonMC': 'BTC',  # 'MC|BTC|BC|NGC|ChC',
        'LBC': 'BC',
        'NBC': 'BC',
        'SBC': 'BC',
        'BTC': 'BTC',
        'DBC': 'BTC',
        'BP': 'BTC',
    }

    dict_common_etype = {
        # 'bAC': 'nFS', 'bIR': 'nFS', 'bNAC': 'nFS', 'cAC': 'nFS', 'cIR': 'nFS', 'bSTUT': 'nFS',
        # 'dNAC': 'FS', 'cNAC': 'FS', 'dSTUT': 'FS', 'cSTUT': 'FS',
        'bAC': 'nFS', 'bIR': 'nFS', 'bNAC': 'nFS', 'cAC': 'nFS', 'cIR': 'nFS', 'cNAC': 'nFS',
        'dNAC': 'FS', 'bSTUT': 'FS', 'dSTUT': 'FS', 'cSTUT': 'FS',
        'ME_Inh_1': 'IR',
        'ME_Inh_2': 'IR',
        'ME_Inh_3': 'IR',
        'ME_Inh_4': 'IR',
        'ME_Inh_5': 'IR',
        'ME_Inh_6': 'FS',
        'ME_Inh_7': 'FS',
        'ME_Inh_8': 'FS',
        'ME_Inh_9': 'FS',
        'ME_Inh_10': 'FS',
        'ME_Inh_11': 'FS',
        'ME_Inh_12': 'FS',
        'ME_Inh_13': 'FS',
        'ME_Inh_14': 'FS',
        'ME_Inh_15': 'Adapt.',
        'ME_Inh_16': 'FS',
        'ME_Inh_17': 'RS',
        'ME_Inh_18': 'IR',
        'ME_Inh_19': 'RS',
        'ME_Inh_20': 'RS',
        'ME_Inh_21': 'FS',
        'ME_Inh_22': 'Adapt.',
        'ME_Inh_23': 'Adapt.',
        'ME_Inh_24': 'Adapt.',
        'ME_Inh_25': 'Adapt.',
        'ME_Inh_26': 'Adapt.'

    }

    dict_common_layers = {
        'L1': 'L1',
        'L23': 'L2/3/4',
        'L2/3': 'L2/3/4',
        'L4': 'L2/3/4',
        'L5': 'L5/6',
        'L6': 'L5/6',
        'L6a': 'L5/6',
        'L6b': 'L5/6'
    }

    dict_common_fine_layers = {
        'L1': 'L1',
        'L23': 'L23',
        'L2/3': 'L23',
        'L4': 'L4',
        'L5': 'L5',
        'L6': 'L6',
        'L6a': 'L6',
        'L6b': 'L6'
    }

    Gouwens_metype_list_ordered = ['ME_Inh_1', 'ME_Inh_2', 'ME_Inh_3', 'ME_Inh_4', 'ME_Inh_5', 'ME_Inh_6', 'ME_Inh_7',
                                   'ME_Inh_8', 'ME_Inh_9', 'ME_Inh_10', 'ME_Inh_11', 'ME_Inh_12', 'ME_Inh_13', 'ME_Inh_14',
                                   'ME_Inh_15', 'ME_Inh_16', 'ME_Inh_17', 'ME_Inh_18', 'ME_Inh_19', 'ME_Inh_20',
                                   'ME_Inh_21', 'ME_Inh_22', 'ME_Inh_23', 'ME_Inh_24', 'ME_Inh_25', 'ME_Inh_26']

    BBP_data_no_realign = pd.read_csv('./filtered_datasets/BBP_dataset_filtered_(no_realign).csv', index_col=0)
    Gouwens_data_no_realign = pd.read_csv('./filtered_datasets/Gouwens_2019_dataset_filtered_(no_realign).csv',
                                          index_col=0)
    BBP_data_no_norm = pd.read_csv('./filtered_datasets/BBP_dataset_filtered_(no_norm).csv', index_col=0)
    Gouwens_data_no_norm = pd.read_csv('./filtered_datasets/Gouwens_2019_dataset_filtered_(no_norm).csv',
                                          index_col=0)

    ## Gouwens meta data for individual cells, e.g. their custom m-, e-, me-types... etc.
    Gouwens_meta = pd.read_excel(io='../downloader/41593_2019_417_MOESM5_ESM.xlsx', sheet_name='Data',
                                 index_col=0)
    ## Layer + driver associated with individual Gouwens cells
    Gouwens_laycre_type = pd.read_csv('./AIBS_lay_cre_type_df.csv', index_col=0)

    for Gouwens_data, BBP_data, title in zip([Gouwens_data_no_realign, Gouwens_data_no_norm],
                                             [BBP_data_no_realign, BBP_data_no_norm],
                                             ["", "_no_norm"]):
        # Gouwens types from metadata
        Gouwens_me_type = [Gouwens_meta['me-type'][int(ID)] for ID in Gouwens_data.index]
        Gouwens_m_type = [Gouwens_meta['m-type'][int(ID)] for ID in Gouwens_data.index]
        Gouwens_common_m_type = [dict_common_names[dict_Gouwens_ME[x].split(' ')[0]] for x in Gouwens_me_type]
        Gouwens_e_type = [Gouwens_meta['e-type'][int(ID)] for ID in Gouwens_data.index]
        Gouwens_common_e_type = [dict_common_etype[x] for x in Gouwens_me_type]
        Gouwens_common_me_type = ['_'.join([x, y]) for (x, y) in zip(Gouwens_common_m_type, Gouwens_common_e_type)]
        Gouwens_molID = [
            dict_common_fine_layers[Gouwens_laycre_type['Layer+driverline'][int(ID)].split(' ')[0]] + '_' +
            dict_Gouwens_ME[me].split(' ')[1]
            for ID, me in zip(Gouwens_data.index, Gouwens_me_type)]

        dict_molID = {x: x for x in unique_elements(Gouwens_molID)}
        dict_molID['L1_Lamp5'] = 'L1_Gad2'
        dict_molID['L1_Vip'] = 'L1_Gad2'

        Gouwens_molID_refined = [dict_molID[x] for x in Gouwens_molID]
        Gouwens_marker = [x.split('_')[-1] for x in Gouwens_molID]
        Gouwens_layer = [x.split('_')[0] for x in Gouwens_molID]
        Gouwens_laycre = np.asarray([Gouwens_laycre_type.T[int(x)][0] for x in Gouwens_data.index])

        Gouwens_labels = pd.DataFrame(np.asarray([Gouwens_e_type, Gouwens_common_e_type,
                                                  Gouwens_m_type, Gouwens_common_m_type,
                                                  Gouwens_me_type, Gouwens_common_me_type,
                                                  Gouwens_molID, Gouwens_molID_refined, Gouwens_layer, Gouwens_marker,
                                                  Gouwens_laycre]).T,
                                      index=Gouwens_data.index,
                                      columns=['e-type', 'common e-type', 'm-type',
                                               'common m-type', 'me-type', 'common me-type',
                                               'molID', 'refined molID', 'layer', 'marker', 'layer driver'])

        Gouwens_labels.to_csv("./filtered_datasets/Gouwens_labels" + title + ".csv")

        # BBP type from me combination:
        BBP_metype = np.asarray([x.split('|')[0] for x in BBP_data.index])

        BBP_etype = np.asarray([lbl.split('_')[-1] for lbl in BBP_metype])
        BBP_mtype = np.asarray(['_'.join(lbl.split('_')[:2]) for lbl in BBP_metype])
        BBP_mtype_no_layer = np.asarray([lbl.split('_')[1] for lbl in BBP_metype])
        BBP_common_mtype = np.asarray([dict_common_names[x.split('_')[1]] for x in BBP_mtype])
        BBP_metype_no_layer = np.asarray(['_'.join(lbl.split('_')[1:]) for lbl in BBP_metype])
        BBP_common_etype = np.asarray([dict_common_etype[x] for x in BBP_etype])
        BBP_layer = np.asarray([lbl.split('_')[0] for lbl in BBP_metype])
        BBP_morpho_ids = np.asarray([x.split('|')[-1] for x in BBP_data.index])
        BBP_common_metypes = np.asarray(['_'.join([x, dict_common_etype[y]])
                                         for (x, y) in zip(BBP_common_mtype,
                                                           BBP_etype)])

        BBP_lay_common_mtypes = np.asarray(['_'.join([x, y])
                                            for (x, y) in zip(BBP_layer,
                                                              BBP_common_mtype)])
        BBP_lay_common_etypes = np.asarray(['_'.join([x, y])
                                            for (x, y) in zip(BBP_layer,
                                                              BBP_common_etype)])

        BBP_labels = pd.DataFrame(np.asarray([BBP_etype, BBP_common_etype,
                                              BBP_mtype, BBP_mtype_no_layer, BBP_common_mtype,
                                              BBP_lay_common_mtypes, BBP_lay_common_etypes,
                                              BBP_metype, BBP_metype_no_layer, BBP_common_metypes,
                                              BBP_layer, BBP_morpho_ids]).T,
                                  index=BBP_data.index,
                                  columns=['e-type', 'common e-type',
                                           'm-type', 'm-type no layer', 'common m-type',
                                           'layer common m-type', 'layer common e-type',
                                           'me-type', 'me-type no layer', 'common me-type',
                                           'layer', 'morphology id'])
        BBP_labels.to_csv("./filtered_datasets/BBP_labels" + title + ".csv")

if __name__ == "__main__":
    compute_labels()
