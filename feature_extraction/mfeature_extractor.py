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

from me_types_mapper.morphology_tools import *
import os


def extract_features_AIBS(ID_list, path_raw, path_norm, d_pia_list, nm_morpho_features, angle_list=[], Normalize=True):
    ax_mom = []
    dend_mom = []
    ID_clean = []
    ID_no_morpho = []

    for cell_id in ID_list:
        try:
            dataset = path_raw.split('/')[2]
            print(dataset)
            if 'Gouw' in dataset:
                path_to_morphology = path_raw + 'specimen_' + str(cell_id) + '_reconstruction.swc'
                print(path_to_morphology)
            elif 'pseq' in dataset:
                path_to_morphology = path_raw + str(cell_id) + '_transformed.swc'
            distance_to_the_pia = d_pia_list[cell_id]
            if len(angle_list) != 0:
                angle = angle_list[cell_id] * (np.pi / 180)

            if Normalize:
                if len(angle_list) != 0:
                    m = normalize_morphology(path_to_morphology, distance_to_the_pia, layer=None, layer_sup=None,
                                             animal='Mouse', brain_region='V1', rotation_angle=angle, realign=False)

                else:
                    m = normalize_morphology(path_to_morphology, distance_to_the_pia, layer=None, layer_sup=None,
                                             animal='Mouse', brain_region='V1', rotation_angle=0, realign=False)

            else:
                m = Morphology(path_to_morphology)
            m.write(path_norm + str(cell_id) + '_normalized_aligned.swc')

            dict_ = extract_moments(m, N=5)
            ax_mom.append(dict_['moments axon'])
            dend_mom.append(dict_['moments dendrites'])
            ID_clean.append(cell_id)
        except:
            print('no morphology for cell ', cell_id)
            ID_no_morpho.append(cell_id)

    print('ID clean: ', len(ID_clean))
    print('ID no morpho: ', len(ID_no_morpho))

    moments_data = moments_to_csv(ax_mom, dend_mom, ID_clean, N=3)
    neurom_data = neurom_extractor(path_norm, nm_morpho_features)
    dict_rename = {name: np.float(name.split('_')[0]) for name in neurom_data.index}
    morpho_data_df = pd.concat([moments_data, neurom_data.rename(dict_rename)],
                               axis=1, sort=True)

    return morpho_data_df


def BBP_path_list(path_raw):
    lst = []
    for name in os.listdir(path_raw):
        if ".swc" in name:
            lst.append(path_raw + '/' + name)
    return lst


def extract_features_BBP(path_list, path_norm, meta_data, dict_layer_sup, nm_morpho_features, Normalize=True):
    ax_mom = []
    dend_mom = []
    ID = []
    ID_no_morpho = []

    for path in path_list:
        # try:
        name = path.split('/')[-1].split('.')[0]
        mtype = meta_data[name]
        lay = mtype.split("_")[0]
        ID.append(mtype + '_' + name + ".swc")
        if Normalize:
            m = normalize_morphology(path, distance_to_the_pia=None, layer=lay, layer_sup=dict_layer_sup[lay],
                                     animal='Rat', brain_region='S1', rotation_angle=0, realign=False)

        else:
            m = Morphology(path)
        m.write(path_norm + mtype + '_normalized_aligned_' + name + ".swc")
        dict_ = extract_moments(m, N=5)
        ax_mom.append(dict_['moments axon'])
        dend_mom.append(dict_['moments dendrites'])

    print('ID clean: ', len(ID))
    print('ID no morpho: ', len(ID_no_morpho))
    moments_data = moments_to_csv(ax_mom, dend_mom, ID, N=3)
    neurom_data = neurom_extractor(path_norm, nm_morpho_features)
    dict_rename = {name : name.replace('_normalized_aligned', '') for name in neurom_data.index}
    morpho_data_df = pd.concat([moments_data, neurom_data.rename(dict_rename)],
                               axis=1, sort=True)

    return morpho_data_df


if __name__ == "__main__":
    nm_morpho_features = pd.read_csv('./NeuroM_morpho_features.csv', index_col=0)['mfeatures'].values

    dict_moment_names = {}
    i = 0
    for neur in ["axon", "dendrites"]:
        for lay in ["L1", "L23", "L4", "L5", "L6"]:
            for mom in ["0", "1x", "1y", "2x", "2y"]:
                dict_moment_names["morpho_moment_" + str(i)] = "morpho_moment_" + neur + "_" + lay + "_µ_" + mom
                i += 1

    ### Gouwens dataset
    print('Gouwens dataset')
    Gouw_meta = pd.read_excel(io='../downloader/41593_2019_417_MOESM5_ESM.xlsx', sheet_name='Data', index_col=0)
    mask_Inh = ['Inh' in str(x) for x in Gouw_meta['me-type'].values]
    Gouw_ID = Gouw_meta.index[mask_Inh]
    if not os.path.isdir('../downloader/Gouwens_2019/Normalized_aligned_morphologies/'):
        os.mkdir('../downloader/Gouwens_2019/Normalized_aligned_morphologies/')

    # not normalized
    Gouw_morpho_data_no_norm_df = extract_features_AIBS(Gouw_ID, '../downloader/Gouwens_2019/morphologies/',
                                                '../downloader/Gouwens_2019/Normalized_aligned_morphologies/',
                                                Gouw_meta[mask_Inh]['soma_distance_from_pia'], nm_morpho_features,
                                                angle_list=Gouw_meta[mask_Inh]['upright_angle'], Normalize=False)
    Gouw_morpho_data_no_norm_df = Gouw_morpho_data_no_norm_df.rename(dict_moment_names, axis=1)
    Gouw_morpho_data_no_norm_df.to_csv('./Gouw_morpho(moments_NeuroM)_features_no_norm.csv')
    # normalized
    Gouw_morpho_data_df = extract_features_AIBS(Gouw_ID, '../downloader/Gouwens_2019/morphologies/',
                                                '../downloader/Gouwens_2019/Normalized_aligned_morphologies/',
                                                Gouw_meta[mask_Inh]['soma_distance_from_pia'], nm_morpho_features,
                                                angle_list=Gouw_meta[mask_Inh]['upright_angle'], Normalize=True)
    Gouw_morpho_data_df = Gouw_morpho_data_df.rename(dict_moment_names, axis=1)
    Gouw_morpho_data_df.to_csv('./Gouw_morpho(moments_NeuroM)_features_no_realign.csv')
    print(Gouw_morpho_data_df)

    ### BBP dataset
    print('BBP dataset')

    morpho_metadata_BBP = pd.read_csv('../downloader/BBP/metadata_morphologies.csv',
                                      index_col=["name"])

    dict_layer_sup = {'L1': 'pia', 'L23': 'L1', 'L4': 'L23', 'L5': 'L4', 'L6': 'L5'}
    if not os.path.isdir('../downloader/BBP/BBP_normalized_aligned_morphologies/'):
        os.mkdir('../downloader/BBP/BBP_normalized_aligned_morphologies/')

    bbp_path_list = BBP_path_list('../downloader/BBP/BBP_morphologies/')
    # not normalized first:
    BBP_morpho_data_no_norm_df = extract_features_BBP(bbp_path_list,
                                              '../downloader/BBP/BBP_normalized_aligned_morphologies/',
                                              morpho_metadata_BBP["annotation.hasBody.label"],
                                              dict_layer_sup, nm_morpho_features, Normalize=False)
    BBP_morpho_data_no_norm_df = BBP_morpho_data_no_norm_df.rename(dict_moment_names, axis=1)
    BBP_morpho_data_no_norm_df.to_csv('./BBP_morpho(moments_NeuroM)_features_no_norm.csv')
    # normalized
    BBP_morpho_data_df = extract_features_BBP(bbp_path_list,
                                              '../downloader/BBP/BBP_normalized_aligned_morphologies/',
                                              morpho_metadata_BBP["annotation.hasBody.label"],
                                              dict_layer_sup, nm_morpho_features, Normalize=True)
    BBP_morpho_data_df = BBP_morpho_data_df.rename(dict_moment_names, axis=1)
    BBP_morpho_data_df.to_csv('./BBP_morpho(moments_NeuroM)_features_no_realign.csv')
    print(BBP_morpho_data_df)
