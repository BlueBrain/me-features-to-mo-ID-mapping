from me_types_mapper.mapper.coclustering_functions import *
from me_types_mapper.mapper.R_values import *
from sklearn.decomposition import PCA
import os


def plot_PC(ax, X_plt, labels_dataset):
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)

    for clr, lbl in zip(["r", "b"], unique_elements(labels_dataset)):
        msk = [l == lbl for l in labels_dataset]
        ax.scatter(X_plt.values[:, 0][msk], X_plt.values[:, 1][msk],
                   c=clr, alpha=0.25)
    for clr, lbl in zip(["darkred", "darkblue"], unique_elements(labels_dataset)):
        msk = [l == lbl for l in labels_dataset]
        centroid = np.mean(X_plt[msk], axis=0)
        #         dist_ = [scipy.spatial.distance.euclidean(np.mean(X_plt[msk],axis=0), X_plt[msk].T[idx])
        #                  for idx in X_plt[msk].index]
        #         radius = np.mean(dist_)
        #         ax.add_artist(plt.Circle((centroid[0], centroid[1]), radius, color=clr, alpha=0.1))
        radius = np.std(X_plt[msk], axis=0)
        ax.add_artist(matplotlib.patches.Ellipse((centroid[0], centroid[1]),
                                                 4 * radius[0],
                                                 4 * radius[1],
                                                 color=clr,
                                                 alpha=0.2))
        ax.plot(centroid[0], centroid[1],
                "s", c=clr, markersize=10, alpha=0.75)
    return ax


def prepare_data_matrices(X_ephys_Gouwens_no_norm, X_morpho_Gouwens_no_norm, X_ephys_Gouwens_raw, X_morpho_Gouwens_raw,
                          X_ephys_Gouwens_z, X_morpho_Gouwens_z,
                          X_ephys_BBP_no_norm, X_morpho_BBP_no_norm, X_ephys_BBP_raw, X_morpho_BBP_raw,
                          X_ephys_BBP_z, X_morpho_BBP_z):

    Xe_no_norm_pca = Combine_data(X_ephys_Gouwens_no_norm, X_ephys_BBP_no_norm)
    Xm_no_norm_pca = Combine_data(X_morpho_Gouwens_no_norm, X_morpho_BBP_no_norm)
    Xme_no_norm = pd.concat([Xe_no_norm_pca, Xm_no_norm_pca], axis=1)
    Xme_no_norm_pca = pd.DataFrame(PCA().fit_transform(Xme_no_norm.values), index=Xme_no_norm.index)

    Xe_raw = pd.concat([X_ephys_Gouwens_raw, X_ephys_BBP_raw], axis=0)
    Xm_raw = pd.concat([X_morpho_Gouwens_raw, X_morpho_BBP_raw], axis=0)
    Xme_raw = pd.concat([Xe_raw, Xm_raw], axis=1)

    Xe_pca = Combine_data(X_ephys_Gouwens_raw, X_ephys_BBP_raw)
    Xm_pca = Combine_data(X_morpho_Gouwens_raw, X_morpho_BBP_raw)
    Xme_pca = pd.DataFrame(PCA().fit_transform(Xme_raw.values), index=Xme_raw.index)

    Xe_z = pd.concat([X_ephys_Gouwens_z, X_ephys_BBP_z], axis=0)
    Xm_z = pd.concat([X_morpho_Gouwens_z, X_morpho_BBP_z], axis=0)
    Xme_z = pd.concat([Xe_z, Xm_z], axis=1)

    Xe_zpca = Combine_data(X_ephys_Gouwens_z, X_ephys_BBP_z)
    Xm_zpca = Combine_data(X_morpho_Gouwens_z, X_morpho_BBP_z)
    Xme_zpca = pd.DataFrame(PCA().fit_transform(Xme_z.values), index=Xme_z.index)

    data = [[Xe_no_norm_pca, Xm_no_norm_pca, Xme_no_norm_pca],
            [Xe_pca, Xm_pca, Xme_pca],
            [Xe_zpca, Xm_zpca, Xme_zpca],
            ]

    return data


if __name__ == "__main__":

    BBP_data = pd.read_csv('../feature_selection/filtered_datasets/BBP_dataset_filtered_(no_realign).csv', index_col=0)
    BBP_data_no_norm = pd.read_csv('../feature_selection/filtered_datasets/BBP_dataset_filtered_(no_norm).csv',
                                   index_col=0)
    BBP_labels = pd.read_csv('../feature_selection/filtered_datasets/BBP_labels.csv', index_col=0)
    BBP_labels_no_norm = pd.read_csv('../feature_selection/filtered_datasets/BBP_labels_no_norm.csv', index_col=0)
    Gouwens_data = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_2019_dataset_filtered_(no_realign).csv',
                               index_col=0)
    Gouwens_data_no_norm = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_2019_dataset_filtered_(no_norm).csv',
                                       index_col=0)
    Gouwens_labels = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_labels.csv', index_col=0)
    Gouwens_labels_no_norm = pd.read_csv('../feature_selection/filtered_datasets/Gouwens_labels_no_norm.csv',
                                         index_col=0)

    # Mask to sort out L1 neurons
    msk_L1out_aibs = np.asarray([x != 'L1' for x in Gouwens_labels['layer']])
    msk_L1out_aibs_no_norm = np.asarray([x != 'L1' for x in Gouwens_labels_no_norm['layer']])
    msk_L1out_bbp = np.asarray([x != 'L1' for x in BBP_labels['layer']])
    msk_L1out_bbp_no_norm = np.asarray([x != 'L1' for x in BBP_labels_no_norm['layer']])

    BBP_data = BBP_data[msk_L1out_bbp]
    BBP_data_no_norm = BBP_data_no_norm[msk_L1out_bbp_no_norm]
    BBP_labels = BBP_labels[msk_L1out_bbp]
    BBP_labels_no_norm = BBP_labels_no_norm[msk_L1out_bbp_no_norm]
    Gouwens_data = Gouwens_data[msk_L1out_aibs]
    Gouwens_data_no_norm = Gouwens_data_no_norm[msk_L1out_aibs_no_norm]
    Gouwens_labels = Gouwens_labels[msk_L1out_aibs]
    Gouwens_labels_no_norm = Gouwens_labels_no_norm[msk_L1out_aibs_no_norm]

    msk_morpho = np.asarray(['morpho' in name for name in Gouwens_data.columns])
    msk_ephys = ~msk_morpho

    if not os.path.exists("validation_results"):
        os.mkdir("validation_results")

    # Data preprocessing
    ##Gouwens
    X_ephys_Gouwens_no_norm, X_morpho_Gouwens_no_norm = Split_raw(Gouwens_data_no_norm, msk_ephys, msk_morpho)
    X_ephys_Gouwens_raw, X_morpho_Gouwens_raw = Split_raw(Gouwens_data, msk_ephys, msk_morpho)
    X_ephys_Gouwens_z, X_morpho_Gouwens_z = Z_scale(Gouwens_data, msk_ephys, msk_morpho)

    ##BBP
    X_ephys_BBP_no_norm, X_morpho_BBP_no_norm = Split_raw(BBP_data_no_norm, msk_ephys, msk_morpho)
    X_ephys_BBP_raw, X_morpho_BBP_raw = Split_raw(BBP_data, msk_ephys, msk_morpho)
    X_ephys_BBP_z, X_morpho_BBP_z = Z_scale(BBP_data, msk_ephys, msk_morpho)

    data = prepare_data_matrices(X_ephys_Gouwens_no_norm, X_morpho_Gouwens_no_norm, X_ephys_Gouwens_raw, X_morpho_Gouwens_raw,
                                 X_ephys_Gouwens_z, X_morpho_Gouwens_z,
                                 X_ephys_BBP_no_norm, X_morpho_BBP_no_norm, X_ephys_BBP_raw, X_morpho_BBP_raw,
                                 X_ephys_BBP_z, X_morpho_BBP_z)

    # R-values
    labels_dataset = np.asarray(
        ['AIBS'] * len(Gouwens_data) + ['BBP'] * len(BBP_data))
    labels_dataset_no_norm = np.asarray(
        ['AIBS'] * len(Gouwens_data_no_norm) + ['BBP'] * len(BBP_data_no_norm))

    R_vals = []

    for i, dat_ in enumerate(data):
        if i == 0:
            labels = labels_dataset_no_norm
        else:
            labels = labels_dataset
        R_vals.append([R_value_U(dat_[0], labels, k=400), #400
                       R_value_U(dat_[1], labels, k=400),
                       R_value_U(dat_[2], labels, k=400)]
                      )

    R_values_res = pd.DataFrame(np.asarray(R_vals),
                                index=['Not Normalised', 'PCA', 'Z-scored + PCA'],
                                columns=['e-features', 'm-features', 'me-features'])

    R_values_res.to_csv('validation_results/R-values_benchmark(Gouwens+BBP)_L1out_1July2021.csv')

    # Plotting

    xlabels_list = ['e-features', 'm-features', 'me-features']

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i, dat_ in enumerate(data):
        if i == 0:
            labels = labels_dataset_no_norm
        else:
            labels = labels_dataset
        for j, dat__ in enumerate(dat_):
            plot_PC(ax[i, j], dat__, labels)

            if i == 0:
                left, bottom, width, height = [(j + 1) * 0.25 + j * 0.025, 0.775, 0.1,
                                               0.05]  # [(j+1)*0.29 - j*0.02, 0.375, 0.07, 0.05]
                ax2 = fig.add_axes([left, bottom, width, height])
                right_side = ax2.spines["right"]
                top_side = ax2.spines["top"]
                right_side.set_visible(False)
                top_side.set_visible(False)
                ax2.bar(1. - 1., R_values_res.T['Not Normalised'][j], color='white', edgecolor='k')
                ax2.bar(1., R_values_res.T['PCA'][j], color='darkgrey')
                ax2.bar(1. + 1., R_values_res.T['Z-scored + PCA'][j], color='grey')
                ax2.set_ylim([0., 1.])
                ax2.set_xticks([.0, 1., 2.])
                ax2.set_xticklabels(['not norm.', 'norm.', 'Z-scaled'], fontsize=12, rotation=30)
                ax2.set_title('R-values', fontsize=14)

            if (i, j) == (0, 0):
                ax[i, j].legend(['AIBS', 'BBP'], fontsize=14, loc='upper left')
            if i == 2:
                ax[i, j].set_xlabel('PC_1', fontsize=16)
            if j == 0:
                ax[i, j].set_ylabel('PC_2', fontsize=16)
    plt.savefig("./validation_results/dataset_overlap_figure.pdf", format="pdf")

