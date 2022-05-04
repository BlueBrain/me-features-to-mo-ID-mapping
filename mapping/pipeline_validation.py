# This file is part of me-features-to-mol-ID-mapping.
#
#
# Copyright © 2021-2022 Blue Brain Project/EPFL
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

import os

from me_types_mapper.mapper.coclustering_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold

### used balanced accuracy instead of accuracy, check if results changed a lot?
def auto_validation_v2(data, msk_ephys_, msk_morpho_, lbls, alpha_list=np.arange(.5, 1.5, 0.05),
                       ratio=.1, N_iter=3, d_opt=None, hca_method='ward'):
    alpha_ = []
    prec_ = []
    acc_ = []
    confmat_ = []

    for n_iter in range(N_iter):
        (X_train, X_test,
         y_train, y_test) = train_test_split(data, lbls, test_size=ratio)

        print('test size: ', len(y_test) / (len(y_test) + len(y_train)))

        label_list = np.asarray(y_train.tolist() + [None] * len(y_test))
        msk_train = [True] * len(X_train) + [False] * len(X_test)
        msk_test = [False] * len(X_train) + [True] * len(X_test)

        percentage_labels_mean_list, integral_vals, alphax, fig_alpha = alpha_opt_v3(X_train, X_test,
                                                                                     msk_ephys_, msk_morpho_,
                                                                                     alpha_list, label_list,
                                                                                     msk_set=msk_train,
                                                                                     hca_method='ward')
        alpha_.append(alphax)

        X_df = preprocess_data(X_train, X_test, msk_ephys_, msk_morpho_, alphax)

        X1_df = X_df[msk_train]
        X2_df = X_df[msk_test]

        label_lst = np.asarray(y_train.tolist() + ['no_label'] * len(y_test))
        label_lst2 = np.asarray(['no_label'] * len(y_train) + y_test.tolist())

        cluster_ref, fig_dopt = forming_clusters(X1_df, X2_df, labels=label_lst, msk_set=msk_train,
                                                 d_opt=d_opt,
                                                 hca_method=hca_method)

        map_, c1, c2, dict_cluster_label = mapping(label_lst, label_lst2, cluster_ref)

        msk_ = np.asarray([False] * len(y_train) + [True] * len(y_test))
        pred_clstr = np.asarray(cluster_ref)[msk_]

        pred_label = np.asarray([dict_cluster_label[clstr] for clstr in pred_clstr])
        msk_pred = [x != 'no_prediction' for x in pred_label]

        prec_.append(precision_score(np.asarray(y_test)[msk_pred],
                                     np.asarray(pred_label)[msk_pred], average='weighted'))

        acc_.append(balanced_accuracy_score(np.asarray(y_test)[msk_pred], np.asarray(pred_label)[msk_pred]))

        confmat_.append(confusion_matrix(np.asarray(y_test)[msk_pred], np.asarray(pred_label)[msk_pred]))

    return map_, c1, c2, alpha_, prec_, acc_, confmat_, cluster_ref


def auto_validation_v3(data, msk_ephys_, msk_morpho_, lbls, alpha_list=np.arange(.5, 1.5, 0.05),
                       N_iter=3, d_opt=None, hca_method='ward'):

    alpha_ = []
    prec_ = []
    acc_ = []
    confmat_ = []
    pred_ratio_ = []

    kf = KFold(n_splits=N_iter, shuffle=True)

    for train_index, test_index in kf.split(data.values):
        X_train, X_test = (pd.DataFrame(data.values[train_index],
                                        columns=data.columns,
                                        index=data.index[train_index]),
                           pd.DataFrame(data.values[test_index],
                                        columns=data.columns,
                                        index=data.index[test_index]))
        y_train, y_test = lbls[train_index], lbls[test_index]

        print('test size: ', len(y_test) / (len(y_test) + len(y_train)))

        label_list = np.asarray(y_train.tolist() + [None] * len(y_test))
        msk_train = [True] * len(X_train) + [False] * len(X_test)
        msk_test = [False] * len(X_train) + [True] * len(X_test)

        percentage_labels_mean_list, integral_vals, alphax, fig_alpha = alpha_opt_v3(X_train, X_test,
                                                                          msk_ephys_, msk_morpho_,
                                                                          alpha_list, label_list,
                                                                          msk_set=msk_train,
                                                                          hca_method='ward')
        alpha_.append(alphax)

        X_df = preprocess_data(X_train, X_test, msk_ephys_, msk_morpho_, alphax)

        X1_df = X_df[msk_train]
        X2_df = X_df[msk_test]

        label_lst = np.asarray(y_train.tolist() + ['no_label'] * len(y_test))
        label_lst2 = np.asarray(['no_label'] * len(y_train) + y_test.tolist())

        cluster_ref, fig_d_opt = forming_clusters(X1_df, X2_df, labels=label_lst, msk_set=msk_train,
                                                  d_opt=d_opt,
                                                  hca_method=hca_method)

        map_, c1, c2, dict_cluster_label = mapping(label_lst, label_lst2, cluster_ref)

        msk_ = np.asarray([False] * len(y_train) + [True] * len(y_test))
        pred_clstr = np.asarray(cluster_ref)[msk_]

        pred_label = np.asarray([dict_cluster_label[clstr] for clstr in pred_clstr])
        msk_pred = [x != 'no_prediction' for x in pred_label]

        prec_.append(precision_score(np.asarray(y_test)[msk_pred],
                                     np.asarray(pred_label)[msk_pred], average='weighted'))

        acc_.append(accuracy_score(np.asarray(y_test)[msk_pred], np.asarray(pred_label)[msk_pred]))

        confmat_.append(confusion_matrix(np.asarray(y_test)[msk_pred], np.asarray(pred_label)[msk_pred]))

        pred_ratio_.append(len(np.asarray(y_test)[msk_pred]) / len(y_test))

    return map_, c1, c2, alpha_, prec_, acc_, confmat_, cluster_ref, pred_ratio_

def within_dataset_validation(label_list, dataset_labels, dataset, dataset_name, msk_ephys, msk_morpho):
    """

    Args:
        label_list:
        dataset_labels:
        dataset:
        dataset_name:
        msk_ephys:
        msk_morpho:

    Returns:

    """
    dict_val = {}
    for label in label_list:
        lbls = dataset_labels[label].values
        res = auto_validation_v3(dataset, msk_ephys, msk_morpho, lbls=lbls,
                                 alpha_list=np.arange(0., 1., .1),
                                 N_iter=10)
        dict_val[(dataset_name, label)] = {}
        for s, i in zip(['alpha', 'precision', 'accuracy', 'pred ratio'], [3, 4, 5, -1]):
            dict_val[(dataset_name, label)][(s, "mean")] = np.mean(res[i])
            dict_val[(dataset_name, label)][(s, "std")] = np.std(res[i])

        dict_val[(dataset_name, label)][("chance level", "mean")] = np.mean(count_elements(lbls) /
                                                                         np.sum(count_elements(lbls)))
        dict_val[(dataset_name, label)][("chance level", "std")] = np.median(count_elements(lbls) /
                                                                          np.sum(count_elements(lbls)))
    return pd.DataFrame(dict_val)


def cross_dataset_validation(data_ref, data_test, msk_dataset_ref, msk_dataset_test, msk_ephys, msk_morpho, lbls):

    (alpha_opt, map_val, c1_val, c2_val,
     dict_cluster_label_val, cluster_ref_val, fig_alpha, fig_d_opt) = cross_predictions_v2(data_ref,
                                                                                data_test,
                                                                                msk_ephys, msk_morpho,
                                                                                lbls,
                                                                                alpha_list_=np.arange(0.0, 1., .1),
                                                                                d_opt=None,
                                                                                hca_method='ward')

    pred_type = np.asarray([dict_cluster_label_val[c] for c in cluster_ref_val[msk_dataset_test]])
    msk_preds = [x not in ['no_prediction', 'BTC|BC|NGC|ChC', 'MC|BTC|BC|NGC|ChC'] for x in pred_type]
    # msk_preds = [x != 'no_prediction' for x in pred_type]

    dict_cross_val = {}
    dict_cross_val["alpha"] = alpha_opt
    dict_cross_val["precision"] = precision_score(pred_type[msk_preds], lbls[msk_dataset_test][msk_preds], average='weighted')
    dict_cross_val["accuracy"] = balanced_accuracy_score(pred_type[msk_preds], lbls[msk_dataset_test][msk_preds])
    dict_cross_val["pred ratio"] = len(pred_type[msk_preds]) / len(pred_type)
    dict_cross_val["chance level (mean)"] = np.mean(
        count_elements(lbls[msk_dataset_ref]) / np.sum(count_elements(lbls[msk_dataset_ref])))
    dict_cross_val["chance level (median)"] = np.median(
        count_elements(lbls[msk_dataset_ref]) / np.sum(count_elements(lbls[msk_dataset_ref])))

    return dict_cross_val


if __name__ == "__main__":

    BBP_data = pd.read_csv('../feature_selection/filtered_datasets/BBP_dataset_filtered_(no_realign).csv', index_col=0)
    BBP_labels = pd.read_csv('../feature_selection/filtered_datasets/BBP_labels.csv', index_col=0)
    Gouwens_data = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_2019_dataset_filtered_(no_realign).csv',
                               index_col=0)
    Gouwens_labels = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_labels.csv', index_col=0)

    # Mask to sort out L1 neurons
    msk_L1out_aibs = np.asarray([x != 'L1' for x in Gouwens_labels['layer']])
    msk_L1out_bbp = np.asarray([x != 'L1' for x in BBP_labels['layer']])

    BBP_data = BBP_data[msk_L1out_bbp]
    BBP_labels = BBP_labels[msk_L1out_bbp]
    Gouwens_data = Gouwens_data[msk_L1out_aibs]
    Gouwens_labels = Gouwens_labels[msk_L1out_aibs]

    msk_morpho = np.asarray(['morpho' in name for name in Gouwens_data.columns])
    msk_ephys = ~msk_morpho

    if not os.path.exists("validation_results"):
        os.mkdir("validation_results")

    # within dataset validation
    Gouwens_validation_df = within_dataset_validation(["marker", "common m-type", "layer"],
                                                      Gouwens_labels, Gouwens_data, "Gouwens", msk_ephys, msk_morpho)
    BBP_validation_df = within_dataset_validation(["common m-type", "layer"],
                                                  BBP_labels, BBP_data, "BBP", msk_ephys, msk_morpho)

    Cross_validation_train_Gouwens_dict = {}
    Cross_validation_train_BBP_dict = {}
    for label_name in ["common m-type", "layer"]:
        lbls = np.asarray([x for x in Gouwens_labels[label_name]] + [x for x in BBP_labels[label_name]])
        msk_Gouw = [True] * len(Gouwens_data) + [False] * len(BBP_data)
        msk_BBP = [False] * len(Gouwens_data) + [True] * len(BBP_data)

        lbls_reverse = np.asarray([x for x in BBP_labels[label_name]] + [x for x in Gouwens_labels[label_name]])
        msk_BBP_reverse = [True] * len(BBP_data) + [False] * len(Gouwens_data)
        msk_Gouw_reverse = [False] * len(BBP_data) + [True] * len(Gouwens_data)


        Cross_validation_train_Gouwens_dict[("training on Gouwens", label_name)] = cross_dataset_validation(
            Gouwens_data, BBP_data,
            msk_Gouw, msk_BBP,
            msk_ephys, msk_morpho,
            lbls)
        Cross_validation_train_BBP_dict[("training on BBP", label_name)] = cross_dataset_validation(
            BBP_data, Gouwens_data,
            msk_BBP_reverse, msk_Gouw_reverse,
            msk_ephys, msk_morpho,
            lbls_reverse)

    Cross_validation_train_Gouwens_df = pd.DataFrame(Cross_validation_train_Gouwens_dict)
    Cross_validation_train_BBP_df = pd.DataFrame(Cross_validation_train_BBP_dict)

    Gouwens_validation_df.to_csv("./validation_results/Gouwens_within_dataset_validation.csv")
    BBP_validation_df.to_csv("./validation_results/BBP_within_dataset_validation.csv")
    Cross_validation_train_Gouwens_df.to_csv("./validation_results/Cross_dataset_validation_train_on_Gouwens.csv")
    Cross_validation_train_BBP_df.to_csv("./validation_results/Cross_dataset_validation_train_on_BBP.csv")

    print("auto val Gouwens", Gouwens_validation_df)
    print("auto val BBP", BBP_validation_df)
    print("cross val Gouwens", Cross_validation_train_Gouwens_df)
    print("cross val BBP", Cross_validation_train_BBP_df)