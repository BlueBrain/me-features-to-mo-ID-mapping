import os
import json
import random
from me_types_mapper.feature_selection_tools import *

def prepare_BBP_efeatures_dataframe(path_efeat='../feature_extraction/BBP_efeatures/'):

    frames = []
    frames_std = []
    unique_efeat = []
    prot = []

    for etype in os.listdir(path_efeat):
        with open(path_efeat + etype + '/features.json', 'r') as f:
            efeat_ = json.load(f)
        key_list = np.asarray([k for k in efeat_.keys()])
        # msk_tmp = [k.startswith('APWaveform_') for k in efeat_.keys()]
        names = []
        values = []
        std = []
        for k in key_list:
            if 'Rest' in k:
                for efeat in efeat_[k]['soma']:
                    k_name = 'step_' + k.split('_')[-1]
                    prot.append(k.split('_')[-1])
                    names.append(efeat['feature'] + '|' + k_name)
                    values.append(efeat['val'][0])
                    std.append(efeat['val'][1])
                    unique_efeat.append(efeat['feature'] + '|' + k_name)

        frames.append(pd.DataFrame(values, index=names, columns=[etype]))
        frames_std.append(pd.DataFrame(std, index=names, columns=[etype]))

    etype_data = pd.concat(frames, axis=1, sort=False, join='outer').T
    etype_data_std = pd.concat(frames_std, axis=1, sort=False, join='outer').T

    return etype_data, etype_data_std

def merge_BBP_m_e_features(BBP_mdata, etype_data, etype_data_std, existing_inh_models):


    index_list = []
    frame = []
    for mtype in BBP_mdata.index:
        m_lbl = '_'.join(mtype.split('_')[:2])
        cell_id = '_'.join(mtype.split('_')[2:])
        for etype in etype_data.index:
            cell_lbl = m_lbl + '_' + etype
            if cell_lbl in existing_inh_models:

                e_feat = [random.gauss(etype_data.T[etype][ef], etype_data_std.T[etype][ef]) for ef in
                          etype_data.columns]
                frame.append(e_feat + BBP_mdata.T[mtype].tolist())
                index_list.append(m_lbl + '_' + etype + '|' + cell_id)
    me_data_df = pd.DataFrame(np.asarray(frame),
                              index=index_list,
                              columns=etype_data.columns.tolist() + BBP_mdata.columns.tolist())

    return me_data_df

def prepare_gouwens_e_dict(path_to_gouw_ephys_features):

    gouw_e_dict = {}
    for dir in os.listdir(path_to_gouw_ephys_features):
        if dir.startswith("specimen"):
            cell_id = np.float(dir.split("_")[1])
            with open(path_to_gouw_ephys_features + dir + "/features.json", 'r') as f:
                cell_e_dict = json.load(f)

                gouw_e_dict[cell_id] = cell_e_dict

    return gouw_e_dict

def convert_edict_to_dataframe(edict):

    pd_dfs = []
    for cell in edict:
        values_vec = []
        column_names = []
        for protocol in edict[cell]:
            for dict_tmp in edict[cell][protocol]["soma"]:
                column_names.append(dict_tmp["feature"] + "|" + protocol)
                values_vec.append(dict_tmp["val"][0])
        pd_dfs.append(pd.DataFrame(values_vec, index=column_names, columns=[cell]))

    return pd.concat(pd_dfs, axis=1).T

def select_me_features(BBP_etype_data, BBP_etype_data_std, BBP_morpho_data, BBP_existing_inh_models,
                       Gouw_e_dict, Gouw_morpho_data, Gouw_lbls):

    # prepare Gouwens_dataset
    Gouw_e_df = convert_edict_to_dataframe(Gouw_e_dict)
    Gouw_me_data = pd.concat([Gouw_e_df, Gouw_morpho_data], axis=1)
    dict_rename = {idx: int(idx) for idx in Gouw_me_data.index}
    Gouw_me_data = Gouw_me_data.rename(dict_rename)
    Gouw_id_list = Gouw_me_data.index.tolist()
    Gouw_me_data = Gouw_me_data.dropna(axis=0, how='all')
    print("Gouwens data unfiltered ", Gouw_me_data)


    # prepare BBP_dataset
    BBP_me_data = merge_BBP_m_e_features(BBP_morpho_data, BBP_etype_data, BBP_etype_data_std, BBP_existing_inh_models)
    BBP_id_list = BBP_me_data.index.tolist()
    BBP_labels = pd.DataFrame(np.asarray([[BBP_cell.split('|')[0],
                                           '_'.join(BBP_cell.split('|')[0].split('_')[:2]),
                                           BBP_cell.split('|')[0].split('_')[-1],
                                           BBP_cell.split('|')[0].split('_')[0]]
                                          for BBP_cell in BBP_me_data.index]),
                              columns=["me-type", "m-type", "e-type", "layer"],
                              index=BBP_me_data.index)
    print("BBP data unfiltered ", BBP_me_data)

    # plot Knowledge graph
    G = instantiate_KG()

    # Gouw_id_list = []
    # G, Gouw_id_list = add_Gouwens_dataset_to_KG(G, Gouw_e_dict, Gouw_morpho_data, Gouw_lbls, Gouw_id_list)
    G = add_dataset_to_KG(G, Gouw_me_data, Gouw_lbls, "Gouwens_2019", "k", "blue")

    # G = add_BBP_gaussian_dataset_to_KG(G, BBP_me_data)
    G = add_dataset_to_KG(G, BBP_me_data, BBP_labels, "BBP", "grey", "lightblue", label_name_list=["me-type", "m-type", "e-type", "layer"])

    # nodes_to_remove = []
    # for x in G.nodes:
    #     try:
    #         G.nodes[x]['color']
    #     except:
    #         nodes_to_remove.append(x)
    # for x in nodes_to_remove:
    #     G.remove_node(x)

    plot_KG(G, show_plot=False)

    All_id_list = Gouw_id_list + BBP_id_list
    list_id_list = [Gouw_id_list, BBP_id_list, All_id_list]
    dataset_names = ['Gouwens_2019', 'BBP', 'all']
    kept_features = KG_analysis_shared_features(G, list_id_list, dataset_names, 'all',
                                                threshold=0.)

    # print("kept features", kept_features)

    efeat_list = [x for x in nx.all_neighbors(G, 'efeature')]
    mfeat_list = [x for x in nx.all_neighbors(G, 'mfeature')]
    # print("e-feat x m-feat", len(efeat_list), len(mfeat_list))


    step_list = ['step_150', 'step_175', 'step_125', 'step_100']

    Gouw_feat_data = kept_features.T[efeat_list].T['Gouwens_2019']
    msk_Gouw_features = (Gouw_feat_data >= 0.).values
    Gouw_feat = Gouw_feat_data[msk_Gouw_features].index

    efeat_noprot_list = unique_elements(np.asarray([efeat.split('|')[0] for efeat in Gouw_feat]))
    print("making Gouwens 2019 dataset")
    # Gouwens_dataset = build_Gouwens_dataset(Gouw_e_dict, Gouw_morpho_data, step_list,
    #                                         Gouw_id_list, efeat_noprot_list, mfeat_list)
    Gouwens_dataset = build_dataset(Gouw_me_data, step_list, Gouw_id_list, efeat_noprot_list, mfeat_list)

    print("Gouwens ", Gouwens_dataset)



    print("making BBP dataset")
    # BBP_Gauss_dataset = build_dataset(BBP_me_data, step_list, BBP_id_list, efeat_noprot_list, mfeat_list)
    BBP_Gauss_dataset = build_dataset(BBP_me_data, step_list, BBP_id_list, efeat_noprot_list, mfeat_list)

    print("BBP ", BBP_Gauss_dataset)

    dset_dict = {
        'Gouwens_2019': Gouwens_dataset,
        'BBP': BBP_Gauss_dataset
    }

    feature_coverage_ref = ref_feature_coverage(dset_dict)
    threshold_values = np.arange(0, 1.0, .01)
    features_coverage, datasets_coverage = Datasets_analysis_shared_features(threshold_values, feature_coverage_ref,
                                                                             dset_dict, list_id_list)
    selected_features, incomplete_samples = shared_features(0.7, feature_coverage_ref, dset_dict)#0.9
    print("selected", len(selected_features), "shared features")
    print(selected_features)
    dataset_filtered = {}

    for dset_name in incomplete_samples.keys():
        msk_cells = [x not in incomplete_samples[dset_name] for x in dset_dict[dset_name].index]
        dataset_filtered[dset_name] = dset_dict[dset_name][msk_cells][selected_features]

    return dataset_filtered


if __name__ == "__main__":

    # load BBP data
    BBP_etype_data, BBP_etype_data_std = prepare_BBP_efeatures_dataframe()
    BBP_morpho_data = pd.read_csv('../feature_extraction/BBP_morpho(moments_NeuroM)_features_no_realign.csv',
                                  index_col=0)
    BBP_morpho_data_no_norm = pd.read_csv('../feature_extraction/BBP_morpho(moments_NeuroM)_features_no_norm.csv',
                                  index_col=0)
    BBP_existing_inh_models = pd.read_csv('existing_inh_models.csv')['0'].values

    # load Gouwens data
    Gouw_lbls = pd.read_csv('./AIBS_labels.csv', index_col=0)
    Gouw_labels = pd.DataFrame(np.asarray([[Gouw_lbls["me-type"][cell],
                                            Gouw_lbls["m-type"][cell],
                                            Gouw_lbls["e-type"][cell],
                                            Gouw_lbls["molecularID"][cell].split("_")[0],
                                            Gouw_lbls["molecularID"][cell].split("_")[1]]
                                            for cell in Gouw_lbls.index]),
                               columns=["me-type", "m-type", "e-type", "layer", "marker"],
                               index=Gouw_lbls.index)

    print("Gouw_lbls", Gouw_lbls)

    print("preparing Gouwens ephys data...")
    Gouw_e_dict = prepare_gouwens_e_dict("../feature_extraction/Gouwens_efeatures/")
    print("Done!")
    Gouw_morpho_data = pd.read_csv('../feature_extraction/Gouw_morpho(moments_NeuroM)_features_no_realign.csv',
                                   index_col=0)
    Gouw_morpho_data_no_norm = pd.read_csv('../feature_extraction/Gouw_morpho(moments_NeuroM)_features_no_norm.csv',
                                   index_col=0)

    for BBP_m_data, Gouw_m_data, norm_title in zip([BBP_morpho_data, BBP_morpho_data_no_norm],
                                                   [Gouw_morpho_data, Gouw_morpho_data_no_norm],
                                                   ['(no_realign)', '(no_norm)']):
        dataset_filtered = select_me_features(BBP_etype_data, BBP_etype_data_std, BBP_m_data, BBP_existing_inh_models,
                                              Gouw_e_dict, Gouw_m_data, Gouw_labels)

        subtitle = '_dataset_filtered_' + norm_title + '.csv'

        print("saving datasets" + norm_title)
        if not os.path.isdir("./filtered_datasets"):
            os.mkdir("./filtered_datasets")
        for dset_name in dataset_filtered.keys():
            print("keeping", len(dataset_filtered[dset_name]), "cells for", dset_name)
            dataset_filtered[dset_name].to_csv("./filtered_datasets/" + dset_name + subtitle)