from me_types_mapper.mapper.coclustering_functions import *
from me_types_mapper.mapper.R_values import *
import os
import matplotlib.patches as patches

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

    m_labels = np.asarray(Gouwens_labels['m-type'].tolist() + BBP_labels['m-type'].tolist())
    e_labels = np.asarray(Gouwens_labels['e-type'].tolist() + BBP_labels['e-type'].tolist())
    me_labels = np.asarray(Gouwens_labels['me-type'].tolist() + BBP_labels['me-type'].tolist())

    msk_Gouw = [True]*len(Gouwens_data) + [False]*len(BBP_data)
    msk_BBP = [False]*len(Gouwens_data) + [True]*len(BBP_data)

    alpha_list = np.arange(0.0, 1.01, 0.1)

    print('Gouwens m-type')
    clstr_hom_Couw_m = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, m_labels, msk_set= msk_Gouw,
                                    hca_method='ward', plotting=False, cmap_color='Reds')
    print('BBP m-type')
    clstr_hom_BBP_m = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, m_labels, msk_set= msk_BBP,
                                    hca_method='ward', plotting=False, cmap_color='Blues')
    print('Gouwens e-type')
    clstr_hom_Couw_e = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, e_labels, msk_set= msk_Gouw,
                                    hca_method='ward', plotting=False, cmap_color='Reds')
    print('BBP e-type')
    clstr_hom_BBP_e = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, e_labels, msk_set= msk_BBP,
                                    hca_method='ward', plotting=False, cmap_color='Blues')
    print('Gouwens me-type')
    clstr_hom_Couw_me = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, me_labels, msk_set= msk_Gouw,
                                    hca_method='ward', plotting=False, cmap_color='Reds')
    print('BBP me-type')
    clstr_hom_BBP_me = alpha_opt_v3(Gouwens_data, BBP_data, msk_ephys, msk_morpho,
                                    alpha_list, me_labels, msk_set= msk_BBP,
                                    hca_method='ward', plotting=False, cmap_color='Blues')

    # percentage_labels_mean_list, integral_vals, alpha_opt_
    dat_list_meta =[[clstr_hom_Couw_e, clstr_hom_Couw_m, clstr_hom_Couw_me],
                    [clstr_hom_BBP_e, clstr_hom_BBP_m, clstr_hom_BBP_me]]
    color_list = ['Reds', 'Blues']
    title_list = ["Gouwens_alpha_opt.pdf", "BBP_alpha_opt.pdf"]

    alpha_list = np.arange(0.0, 1.01, 0.1)

    for color, dat_list, title in zip(color_list, dat_list_meta, title_list):
        cmap = matplotlib.cm.get_cmap(color)

        fig, ax = plt.subplots(3,2, sharex='col', figsize=(8,8))

        for i,dat in enumerate(dat_list):

            dist_array = dist_array = np.arange(1,101,1)
            coeff_list = alpha_list
            percentage_labels_mean_list = dat[0]
            integral_vals = dat[1]
            alpha_opt_ = dat[2]

            for ax_ in ax.flatten():
                        right_side = ax_.spines["right"]
                        top_side = ax_.spines["top"]
                        right_side.set_visible(False)
                        top_side.set_visible(False)

            for j,percentage_labels_mean in enumerate(percentage_labels_mean_list):
                ax[i,0].plot(dist_array, percentage_labels_mean[0],
                        c=cmap((j+1)/len(percentage_labels_mean_list)))
        #     ax[i,0].set_xlabel('Normalised clustering distance %', fontsize=14)
        #     ax[i,0].set_title('Average cluster homogeneity', fontsize=14)
            ax[i,0].set_ylim([0.,1.])
            ax[i,0].set_xlim([0.,100.])
            ax[i,0].set_xticks([0., 50., 100.])
            ax[i,0].set_xticklabels([0, 50, 100], fontsize=14)
            ax[i,0].set_yticks([0., .5, 1.])
            ax[i,0].set_yticklabels([0, .5, 1], fontsize=14)

            ax[i,1].plot(coeff_list, integral_vals, '-o', c='k', ms=8)
            for j, (x_, y_) in enumerate(zip(coeff_list, integral_vals)):
                ax[i,1].plot(x_, y_, 'o', c=cmap((j+1)/len(coeff_list)), ms=7, clip_on=False)
        #     ax[i,1].set_xlabel('α coefficient', fontsize=14)
        #     ax[i,1].set_title('Integral value', fontsize=14)
            ax[i,1].axvline(x=alpha_opt_, c='k', ls='--')
            ax[i,1].text(alpha_opt_,0.95, 'α = ' + str(round(alpha_opt_, 2)), fontsize=16)
            ax[i,1].set_ylim([0.,1.])
            ax[i,1].set_ylim([0.,2.])
            ax[i,1].set_ylim([0.,1.])
    #         ax[i,1].set_xticks([0, 1, 2, 3, 4, 5])
    #         ax[i,1].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=14)
            ax[i,1].set_xticks([0, 0.5, 1])
            ax[i,1].set_xticklabels([0, 0.5, 1], fontsize=14)
            ax[i,1].set_yticks([0., .5, 1.])
            ax[i,1].set_yticklabels([0, .5, 1], fontsize=14)

        fig.savefig("./validation_results/" + title, format="pdf")

    # R-value dependence on alpha

    R_values = []
    radius = []
    radius_std = []
    labels_dataset = np.asarray(['AIBS'] * len(Gouwens_data) +
                                  ['BBP'] * len(BBP_data))

    for alpha in np.arange(0., 1., 0.05):
        X_super_opt = preprocess_data(Gouwens_data,
                                      BBP_data,
                                      msk_ephys, msk_morpho, alpha)

        R_values.append(R_value_U(X_super_opt, labels_dataset))
        dist_ = [scipy.spatial.distance.euclidean(np.mean(X_super_opt, axis=0), X_super_opt.T[idx])
                 for idx in X_super_opt.index]
        radius.append(np.mean(dist_))
        radius_std.append(np.std(dist_))

    alpha_vs_R = pd.DataFrame(np.asarray([np.arange(0., 1., 0.05), R_values, radius, radius_std]),
                              index=['alpha', 'R-values', 'radius', 'radius std']).T

    alpha_vs_R.to_csv('./validation_results/R-values_VS_alpha(Gouwens+BBP).csv')

    fig, ax = plt.subplots(1, 2, figsize=(8, 2))

    for ax_ in ax:
        right_side = ax_.spines["right"]
        top_side = ax_.spines["top"]
        right_side.set_visible(False)
        top_side.set_visible(False)

    ax[0].plot(alpha_vs_R['alpha'], np.asarray(alpha_vs_R['R-values']), 'k', clip_on=True)
    ax[0].set_xlim([0, 1.])
    ax[0].set_ylim([0., 1.])
    ax[0].set_yticks([0., .5, 1.])
    ax[0].set_yticklabels([0, .5, 1], fontsize=14)

    rect = patches.Rectangle((0., .62), .5, .38, linewidth=1, ls='--', edgecolor='grey', facecolor='none')
    ax[0].add_patch(rect)

    left, bottom, width, height = [0.17, 0.25, 0.15, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(alpha_vs_R['alpha'], np.asarray(alpha_vs_R['R-values']), 'k', clip_on=True)
    ax2.set_xlim([0., .5])
    ax2.set_ylim([.6, 1.])

    ax[1].plot(alpha_vs_R['alpha'], alpha_vs_R['radius'], 'k')
    ax[1].fill_between(alpha_vs_R['alpha'], np.asarray(alpha_vs_R['radius']) - np.asarray(alpha_vs_R['radius std']),
                       np.asarray(alpha_vs_R['radius']) + np.asarray(alpha_vs_R['radius std']), color='grey', alpha=0.2)
    ax[1].set_yticks([0, 5, 10])
    ax[1].set_yticklabels([0, 5, 10], fontsize=14)

    fig.savefig("./validation_results/R_VS_alpha.pdf", format="pdf")
