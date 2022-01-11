from me_types_mapper.mapper.coclustering_functions import *
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import os

def plot_matrix(ax, X , axis_labels=True):
    X = X.div(np.sum(X, axis=0), axis=1).fillna(0)

    ax.imshow(X, vmin=0., vmax=1., cmap='Greys')
    if axis_labels:
        ax.set_xticks(np.arange(len(X.columns)))
        ax.set_xticklabels(X.columns, rotation=90)
        ax.set_yticks(np.arange(len(X.index)))
        ax.set_yticklabels(X.index)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Loop over data dimensions and create text annotations.
    for i in range(len(X.index)):
        for j in range(len(X.columns)):
            if (X.values[i, j] > 0.5):
                text = ax.text(j, i, int(X.values[i, j] * 100),
                               ha="center", va="center", color="w")
            elif (0.1 < X.values[i, j] <= 0.5):
                text = ax.text(j, i, int(X.values[i, j] * 100),
                               ha="center", va="center", color="k")

    return

def plot_column(ax, X):
    #     X = X.div(np.sum(X, axis=0), axis=1).fillna(0)

    ax.imshow(X, cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(X.index)))
    ax.set_yticklabels(X.index)

    # Loop over data dimensions and create text annotations.
    for i in range(len(X.index)):
        if (X.values[i] > 0.5 * np.max(X.values)):
            text = ax.text(0, i, int(X.values[i]),
                           ha="center", va="center", color="w")
        elif (X.values[i] <= 0.5 * np.max(X.values)):
            text = ax.text(0, i, int(X.values[i]),
                           ha="center", va="center", color="k")

    return

def plot_row(ax, X):
    #     X = X.div(np.sum(X, axis=0), axis=1).fillna(0)

    ax.imshow(X, cmap='Greys')
    ax.set_xticks(np.arange(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=90)
    ax.set_yticks([])

    # Loop over data dimensions and create text annotations.
    for i in range(len(X.columns)):
        if (X.values[0][i] > 0.5 * np.max(X.values)):
            text = ax.text(i, 0, int(X.values[0][i]),
                           ha="center", va="center", color="w")
        elif (X.values[0][i] <= 0.5 * np.max(X.values)):
            text = ax.text(i, 0, int(X.values[0][i]),
                           ha="center", va="center", color="k")

    return

def plot_mapping_matrices(c_test, c_ref, P_lbl_test_c, P_c_lbl_ref, P_label_test_label_ref):
    dict_clstr_name = {i + 1: 'Cluster_' + str(i + 1) for i in range(25)}

    fig3 = plt.figure(constrained_layout=True, figsize=(18, 8))
    f3_ax1 = fig3.add_axes([.09, .3, .22, .4])  # [left, bottom, width, height]
    f3_ax2 = fig3.add_axes([.4, .15, .22, .75])
    f3_ax3 = fig3.add_axes([.66, .3, .25, .4])

    f3_ax4 = fig3.add_axes([0.09, 0.3, .22, .05])
    f3_ax5 = fig3.add_axes([0.4, 0.16, .22, .05])

    h1 = (9/19) * (18/8) * 0.22
    f3_ax6 = fig3.add_axes([0.05, (0.4 - h1) * .5 + .3, .03, h1])
    h2 = (19/16) * (18/8) * 0.22
    f3_ax7 = fig3.add_axes([0.36, (0.75 - h2) * .5 + .15, .03, h2])

    plot_matrix(f3_ax1, P_lbl_test_c.rename(dict_clstr_name, axis=0).T, axis_labels=False)
    plot_matrix(f3_ax2, P_c_lbl_ref.rename(dict_clstr_name, axis=0), axis_labels=False)
    plot_matrix(f3_ax3, P_label_test_label_ref)

    plot_row(f3_ax4,
             pd.DataFrame(np.sum(c_test.reindex(P_lbl_test_c.index), axis=1)).rename(dict_clstr_name, axis=0).T
             )
    plot_row(f3_ax5,
             pd.DataFrame(np.sum(c_ref.reindex(P_c_lbl_ref.columns, axis=1))).T
             )

    plot_column(f3_ax6,
                pd.DataFrame(np.sum(c_test.reindex(P_lbl_test_c.columns, axis=1)))
                )
    plot_column(f3_ax7,
                pd.DataFrame(
                    np.sum(c_ref.reindex(P_c_lbl_ref.columns, axis=1).reindex(P_c_lbl_ref.index), axis=1).rename(
                        dict_clstr_name, axis=0))
                )

    # plt.show()

    return fig3

def prepare_counts_for_river_plot(clusters_me_counts_ref, clusters_me_counts_test):
    clstr_order = ['Cluster_' + str(i + 1) for i in range(19)]
    dict_clstr_name = {i + 1: 'Cluster_' + str(i + 1) for i in range(25)}

    clusters_me_map_ref = clusters_me_counts_ref.rename(dict_clstr_name, axis=0).reindex(clstr_order)
    clusters_me_map_test = clusters_me_counts_test.rename(dict_clstr_name, axis=0).reindex(clstr_order)

    clusters_me_map_ref = pd.concat([clusters_me_map_ref.T[['Cluster_1', 'Cluster_3', 'Cluster_6',
                                                            'Cluster_9', 'Cluster_13','Cluster_15',
                                                            'Cluster_17', 'Cluster_18', 'Cluster_19']].sum(axis=1),
                                     clusters_me_map_ref.T[['Cluster_2']],
                                     clusters_me_map_ref.T[['Cluster_4']],
                                     clusters_me_map_ref.T[['Cluster_5']],
                                     clusters_me_map_ref.T[['Cluster_7']],
                                     clusters_me_map_ref.T[['Cluster_8', 'Cluster_10']].sum(axis=1),
                                     clusters_me_map_ref.T[['Cluster_11']],
                                     clusters_me_map_ref.T[['Cluster_12']],
                                     clusters_me_map_ref.T[['Cluster_14']],
                                     clusters_me_map_ref.T[['Cluster_16']]
                                     ],
                                    axis=1).rename({0: 'No match',
                                                    1: 'Cluster_8, 10; (Pvalb)'
                                                    }, axis=1).T

    mtype_order = ['ChC', 'SBC', 'NBC', 'LBC', 'MC', 'BTC', 'BP', 'DBC', 'NGC']
    clusters_me_map_test = clusters_me_map_test.reindex(mtype_order, axis=1)
    clusters_me_map_test = pd.concat([clusters_me_map_test.T[['Cluster_1', 'Cluster_3', 'Cluster_6',
                                                            'Cluster_9', 'Cluster_13','Cluster_15',
                                                            'Cluster_17', 'Cluster_18', 'Cluster_19']].sum(axis=1),
                                     clusters_me_map_test.T[['Cluster_2']],
                                     clusters_me_map_test.T[['Cluster_4']],
                                     clusters_me_map_test.T[['Cluster_5']],
                                     clusters_me_map_test.T[['Cluster_7']],
                                     clusters_me_map_test.T[['Cluster_8', 'Cluster_10']].sum(axis=1),
                                     clusters_me_map_test.T[['Cluster_11']],
                                     clusters_me_map_test.T[['Cluster_12']],
                                     clusters_me_map_test.T[['Cluster_14']],
                                     clusters_me_map_test.T[['Cluster_16']]
                                      ],
                                     axis=1).rename({0: 'No match',
                                                     1: 'Cluster_8, 10; (Pvalb)'
                                                     }, axis=1).T

    clusters_me_map_ref = clusters_me_map_ref.div(np.sum(clusters_me_map_ref, axis=1), axis=0).drop(['No match'],
                                                                                                    axis=0)
    clusters_me_map_test = clusters_me_map_test.div(np.sum(clusters_me_map_test, axis=1), axis=0).drop(['No match'],
                                                                                                       axis=0)

    return clusters_me_map_ref, clusters_me_map_test

def river_plot(clusters_me_counts_ref, clusters_me_counts_test):
    clusters_me_map_ref, clusters_me_map_test = prepare_counts_for_river_plot(clusters_me_counts_ref,
                                                                              clusters_me_counts_test)
    dict_node_color = {
        'L1_neu': 'pink',
        'L1': 'pink',
        'L23': 'orange',
        'L2/3': 'orange',
        'L4': 'lightgreen',
        'L5': 'lightblue',
        'L6': 'purple',
        'L6a': 'purple',
        'L6b': 'purple',

        'BP': 'gold',
        'BTC': 'blue',
        'ChC': 'darkgreen',
        'DAC': 'pink',
        'DBC': 'gold',
        'DLAC': 'pink',
        'HAC': 'pink',
        'LBC': 'green',
        'MC': 'darkblue',
        'NBC': 'green',
        'NGC': 'pink',
        'NGC-DA': 'pink',
        'NGC-SA': 'pink',
        'SBC': 'green',
        'SLAC': 'pink',

        'Pvalb': 'darkgreen',
        'Sst': 'darkblue',
        'Vip': 'gold',
        'Lamp5': 'pink',
        'Serpinf1': 'violet',
        'Sncg': 'lightsalmon',

        'Pvalb, (Cluster': 'grey',
        'Lamp5, (Cluster': 'grey',
        'Vip, (Cluster': 'grey',
        'No match': 'black',

        'ME': 'grey',
        'No': 'black',
        'match': 'black',
        'Aspiny': 'grey',
        'Inh': 'grey',
        'Cluster': 'grey',

        'bAC': 'grey',
        'bIR': 'grey',
        'bNAC': 'grey',
        'bSTUT': 'grey',
        'cAC': 'grey',
        'cIR': 'grey',
        'cNAC': 'grey',
        'cSTUT': 'grey',
        'dNAC': 'grey',
        'dSTUT': 'grey',

        'nan': 'darkgrey'
    }

    N_clust = len(clusters_me_map_test.index)

    cluster_lbl = clusters_me_map_test.index.tolist()

    label_node = clusters_me_map_test.columns.tolist() + cluster_lbl + clusters_me_map_ref.columns.tolist()
    color_node = [dict_node_color[lbl.split('_')[0]] for lbl in label_node]

    values = clusters_me_map_test.T.fillna(0).values.flatten().tolist() + clusters_me_map_ref.fillna(
        0).values.flatten().tolist()

    colors = np.asarray(["pink"] * 6 * N_clust + ["orange"] * 9 * N_clust + ["lightgreen"] * 7 * N_clust +
                        ["lightblue"] * 8 * N_clust + ["purple"] * 8 * N_clust +
                        ["silver"] * (N_clust) * len(clusters_me_map_ref.columns)).flatten()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            # x=[0.1] * len(clusters_me_map_test.columns) + [0.5] * len(clusters_me_map_test.index) + [0.9] * len(
            #     clusters_me_map_ref.columns),
            # y=([(i + 0.1) / len(clusters_me_map_test.columns) for i in range(len(clusters_me_map_test.columns))] +
            #    [(i + 0.1) / len(clusters_me_map_test.index) for i in range(len(clusters_me_map_test.index))] +
            #    [(i + 0.1) / len(clusters_me_map_ref.columns) for i in range(len(clusters_me_map_ref.columns))]),
            thickness=20,
            label=label_node,
            color=color_node
        ),
        link=dict(
            source=np.asarray([[i] * len(clusters_me_map_test.index.tolist()) for i in
                               range(len(clusters_me_map_test.columns.tolist()))]).flatten().tolist() + np.asarray(
                [[i + len(clusters_me_map_test.columns)] * len(clusters_me_map_ref.columns.tolist()) for i in
                 range(len(clusters_me_map_test.index.tolist()))]).flatten().tolist(),

            target=np.asarray([np.arange(len(clusters_me_map_test.index)) + len(clusters_me_map_test.columns)] * len(
                clusters_me_map_test.columns)).flatten().tolist() + np.asarray([np.arange(
                len(clusters_me_map_ref.columns)) + len(clusters_me_map_test.columns) + len(
                clusters_me_map_test.index)] * len(clusters_me_map_test.index)).flatten().tolist(),

            value=values * 100,
            #       color = colors

        ))])

    fig.update_layout(title_text="Cluster mapping", font_size=10)
    # fig.show()

    return fig

def prepare_labels_for_tsne(labels_ref, labels_test):
    lbls_dataset = np.asarray(['BBP'] * len(labels_test) + ['AIBS'] * len(labels_ref)
                              )
    lbls_layer = np.asarray(labels_test['layer'].tolist() +
                            labels_ref['layer'].tolist()
                            )

    lbls_mtype = np.asarray(labels_test['m-type no layer'].tolist() +
                            labels_ref['m-type'].tolist()
                            )
    lbls_etype = np.asarray(labels_test['e-type'].tolist() +
                            labels_ref['e-type'].tolist()
                            )
    lbls_molid = np.asarray(
        ['no mol_ID'] * len(BBP_data) + labels_ref['marker'].tolist()
        )

    lbls_list = [lbls_dataset, lbls_layer, lbls_mtype, lbls_etype, lbls_molid]
    cmap_list = [matplotlib.cm.get_cmap('bwr'),
                 {'L1': 'pink', 'L23': 'orange', 'L4': 'lightgreen', 'L5': 'lightblue', 'L6': 'purple'},
                 (matplotlib.cm.get_cmap('ocean'), matplotlib.cm.get_cmap('autumn')),
                 (matplotlib.cm.get_cmap('winter'), matplotlib.cm.get_cmap('spring')),
                 matplotlib.cm.get_cmap('jet')]
    return lbls_list, cmap_list

def plot_tsne(data_ref, data_test, msk_ephys, msk_morpho,
              labels_ref, labels_test, msk_L1out_ref, msk_L1out_test, alpha_):

    lbls_list, cmap_list = prepare_labels_for_tsne(labels_ref, labels_test)
    X_me_df = preprocess_data(data_test, data_ref,
                              msk_ephys, msk_morpho, alpha_)

    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200, n_iter=1000,
                n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random',
                verbose=0, random_state=None, method='barnes_hut', angle=0.5)

    X_tsne = tsne.fit_transform(X_me_df.values)

    fig, ax = plt.subplots(1, 5, figsize=(15, 3), frameon=False)
    legend_elements = []
    legend_elements2 = []

    msk_aibs = [x == 'AIBS' for x in lbls_list[0]]
    msk_bbp = [x == 'BBP' for x in lbls_list[0]]

    for ax_ in ax:
        for side in ["top", "right"]:
            ax_.spines[side].set_visible(False)

    for j, (lbls_, cmap) in enumerate(zip(lbls_list, cmap_list)):
        if j in [0, 4]:
            for k, lbl in enumerate(unique_elements(lbls_)):
                msk = [l == lbl for l in lbls_]
                ax[j].plot(X_tsne[:, 0][msk], X_tsne[:, 1][msk], 'o',
                           c=cmap((k / (len(unique_elements(lbls_)) - 1)) + .01),
                           alpha=0.25)
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=lbl,
                                              markerfacecolor=cmap((k / (len(unique_elements(lbls_)) - 1)) + .01),
                                              markersize=10))

        elif j in [1]:
            for k, lbl in enumerate(unique_elements(lbls_)):
                msk = [l == lbl for l in lbls_]
                ax[j].plot(X_tsne[:, 0][msk], X_tsne[:, 1][msk], 'o',
                           c=cmap[lbl],
                           alpha=0.25)
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=lbl,
                                              markerfacecolor=cmap[lbl],
                                              markersize=10))

        elif j in [2, 3]:
            for msk_dset, cmap_ in zip([msk_bbp, msk_aibs], cmap):
                for k, lbl in enumerate(unique_elements(lbls_[msk_dset])):
                    msk = [l == lbl for l in lbls_[msk_dset]]
                    ax[j].plot(X_tsne[:, 0][msk_dset][msk], X_tsne[:, 1][msk_dset][msk], 'o',
                               c=cmap_((k / (len(unique_elements(lbls_[msk_dset])) - 1)) + .01),
                               alpha=0.25)
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=lbl,
                                                  markerfacecolor=cmap_(
                                                      (k / (len(unique_elements(lbls_[msk_dset])) - 1)) + .01),
                                                  markersize=10))
        ax[0].set_xlabel('TSNE_1')
        if j == 0:
            ax[j].set_ylabel('TSNE_2')

    fig2, ax2 = plt.subplots(figsize=(6, 6), frameon=False)
    cmap0 = matplotlib.cm.get_cmap('jet')

    cluster_ref_ = cluster_ref[msk_bbp].tolist() + cluster_ref[msk_aibs].tolist()

    for side in ["top", "right"]:
        ax2.spines[side].set_visible(False)
    order = np.argsort(unique_elements(cluster_ref_))
    for k, lbl in enumerate(np.asarray(unique_elements(cluster_ref_))[order]):
        msk = [l == lbl for l in cluster_ref_]
        ax2.plot(X_tsne[:, 0][msk], X_tsne[:, 1][msk], 'o',
                 c=cmap0((k / (len(unique_elements(cluster_ref_)) - 1)) + .01),
                 alpha=0.25)
        legend_elements2.append(Line2D([0], [0], marker='o', color='w', label='cluster_' + str(lbl),
                                       markerfacecolor=cmap0((k / (len(unique_elements(cluster_ref_)) - 1)) + .01),
                                       markersize=10))
    ax2.set_xlabel('TSNE_1', fontsize=16)
    ax2.set_ylabel('TSNE_2', fontsize=16)

    fig_legend, ax_legend = plt.subplots(figsize=(2, 20), frameon=False)
    fig2_legend, ax2_legend = plt.subplots(figsize=(2, 5), frameon=False)

    for side in ["top", "bottom", "right", "left"]:
        ax_legend.spines[side].set_visible(False)
        ax2_legend.spines[side].set_visible(False)

    ax_legend.legend(handles=legend_elements)
    ax2_legend.legend(handles=legend_elements2)

    # plt.show()

    return fig, fig2, fig_legend, fig2_legend

def plot_dataset_coverage(label, labels_test, msk_test, dict_cluster_label, cluster_ref):
    mapped_cell = [dict_cluster_label[x] for x in cluster_ref[msk_test]]
    msk_ = np.asarray([x == 'no_prediction' for x in mapped_cell])

    cts1 = count_elements(labels_test[label][~msk_])  # m-type no layer #e-type
    cts2 = count_elements(labels_test[label][msk_])

    print(cts1)
    print(cts2)
    print(cts1 + cts2)

    fig, ax = plt.subplots(figsize=(8, 4), frameon=False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    plt.bar(cts1.index, (cts1 + cts2).reindex(cts1.index).T.values[0], color='pink', edgecolor='r')
    plt.bar(cts1.index, cts1.T.values[0], color='lightblue', edgecolor='b')
    plt.legend(['no match', 'match'], fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.show()
    return fig

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

    lbls = np.asarray(
        Gouwens_labels['marker'].tolist() +
        BBP_labels['common me-type'].tolist()
    )

    labels_dataset = np.asarray(
        ['AIBS'] * len(Gouwens_data) +
        ['BBP'] * len(BBP_data)
    )

    (alpha_opt, map_, c1, c2,
     dict_cluster_label, cluster_ref, fig_alpha, fig_d_opt) = cross_predictions_v2(Gouwens_data,
                                                                                   BBP_data,
                                                                                   msk_ephys, msk_morpho,
                                                                                   lbls,
                                                                                   alpha_list_=np.arange(0., 1., .05),
                                                                                   d_opt=None,
                                                                                   hca_method='ward')

    if not os.path.exists("figures"):
        os.mkdir("figures")

    # fig_alpha.tight_layout()
    fig_alpha.savefig("./figures/alpha_optimization.pdf", format="pdf")
    # fig_d_opt.tight_layout()
    fig_d_opt.savefig("./figures/clustering_distance_optimization.pdf", format="pdf")

    cell_id_list = Gouwens_data.index.tolist() + BBP_data.index.tolist()
    cluster_ref_dict = {}
    for cell_idx, cell_id in enumerate(cell_id_list):
        cluster_ref_dict[cell_id] = cluster_ref[cell_idx]

    fig_tsne, fig2_tsne, fig_tsne_legend, fig2_tsne_legend = plot_tsne(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                                                       Gouwens_labels, BBP_labels,
                                                                       msk_L1out_aibs, msk_L1out_bbp, alpha_opt)
    fig_tsne.tight_layout()
    fig_tsne_legend.tight_layout()
    fig2_tsne.tight_layout()
    fig2_tsne_legend.tight_layout()
    fig_tsne.savefig("./figures/tsne_labels.pdf", format="pdf")
    fig_tsne_legend.savefig("./figures/labels_legend.pdf", format="pdf")
    fig2_tsne.savefig("./figures/tsne_clusters.pdf", format="pdf")
    fig2_tsne_legend.savefig("./figures/cluster_legend.pdf", format="pdf")

    mask_tmp_BBP = np.asarray([x == 'BBP' for x in labels_dataset])
    mask_tmp_AIBS = np.asarray([x == 'AIBS' for x in labels_dataset])

    etype_coverage_fig = plot_dataset_coverage("e-type", BBP_labels, mask_tmp_BBP,
                                               dict_cluster_label, cluster_ref)
    etype_coverage_fig.tight_layout()
    etype_coverage_fig.savefig("./figures/e_type_coverage.pdf", format="pdf")

    mtype_coverage_fig = plot_dataset_coverage("m-type no layer", BBP_labels, mask_tmp_BBP,
                                               dict_cluster_label, cluster_ref)
    mtype_coverage_fig.tight_layout()
    mtype_coverage_fig.savefig("./figures/m_type_coverage.pdf", format="pdf")

    ### Compute maps

    bbp_label = 'm-type no layer'  # 'm-type no layer'#'common me-type'#'layer common e-type'
    aibs_label = 'refined molID'  # 'molID'#'common me-type'#

    p_maps = compute_probabilistic_maps(BBP_labels, bbp_label, mask_tmp_BBP,
                                        Gouwens_labels, aibs_label, mask_tmp_AIBS,
                                        cluster_ref_dict)

    if not os.path.exists("P_maps"):
        os.mkdir("P_maps")

    p_maps[0].to_csv('./P_maps/Counts_cluster_marker_Gouwens_L1_out.csv')
    p_maps[3].to_csv('./P_maps/Counts_cluster_mtypes_BBP_L1_out.csv')
    p_maps[6].to_csv('./P_maps/P(BBPmtypes_molID)_L1_out.csv')


    mapping_figure = plot_mapping_matrices(p_maps[3], p_maps[0], p_maps[4], p_maps[2], p_maps[6])
    # mapping_figure.tight_layout()
    mapping_figure.savefig("./figures/mapping_matrices.pdf", format="pdf")

    ### Supp. info figures

    p_maps_mkr = compute_probabilistic_maps(BBP_labels, bbp_label, mask_tmp_BBP,
                                            Gouwens_labels, "marker", mask_tmp_AIBS,
                                            cluster_ref_dict)

    river_plot = river_plot(p_maps_mkr[0], p_maps_mkr[3])
    river_plot.write_image("./figures/river_plot.pdf")

    ### Supp. info figures

    p_maps_etype = compute_probabilistic_maps(BBP_labels, "e-type", mask_tmp_BBP,
                                            Gouwens_labels, "common e-type", mask_tmp_AIBS,
                                            cluster_ref_dict)

    p_maps_etype[7].to_csv('./P_maps/P(common_etype_BBPetype).csv')

    p_maps_metype = compute_probabilistic_maps(BBP_labels, "common me-type", mask_tmp_BBP,
                                              Gouwens_labels, "marker", mask_tmp_AIBS,
                                              cluster_ref_dict)

    p_maps_metype[7].to_csv('./P_maps/P(marker_common_metype).csv')
