from itertools import combinations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch 


def build_dko_indicator_matrix(n_guides):

    samples = n_guides**3

    ind_mat = torch.zeros(math.comb(n_guides, 2), samples)
    index_lookup = {}

    for i, combo in enumerate(list(combinations(range(n_guides), 2))):
        index_lookup[combo] = i

    cur_sample = 0

    for i in range(n_guides):

        for j in range(n_guides):

            for k in range(n_guides):

                combo_1 = tuple(sorted((i, j)))
                combo_2 = tuple(sorted((i, k)))
                combo_3 = tuple(sorted((j, k)))

                if combo_1 in index_lookup:
                    ind_mat[index_lookup[combo_1], cur_sample] = 1

                if combo_2 in index_lookup:
                    ind_mat[index_lookup[combo_2], cur_sample] = 1

                if combo_3 in index_lookup:
                    ind_mat[index_lookup[combo_3], cur_sample] = 1

                cur_sample += 1

    return ind_mat


def build_tko_indicator_matrix(n_guides):

    samples = n_guides**3

    ind_mat = torch.zeros(math.comb(n_guides, 3), samples)
    index_lookup = {}

    for i, combo in enumerate(list(combinations(range(n_guides), 3))):
        index_lookup[combo] = i

    cur_sample = 0

    for i in range(n_guides):

        for j in range(n_guides):

            for k in range(n_guides):

                combo = tuple(sorted((i, j, k)))

                if combo in index_lookup:
                    ind_mat[index_lookup[combo], cur_sample] = 1

                cur_sample += 1

    return ind_mat


def build_dko_index_mapping_dict(sko_index_mapping_dict):

    count = 0

    init_dict = {}

    for key_1, val_1 in sko_index_mapping_dict.items():

        for key_2, val_2 in sko_index_mapping_dict.items():

            if key_1 == key_2:
                continue

            dko_str_1 = f"{val_1}-{val_2}"
            dko_str_2 = f"{val_2}-{val_1}"

            if (dko_str_1 not in init_dict) and (dko_str_2 not in init_dict):

                init_dict[dko_str_1] = count
                count += 1

    dko_index_mapping_dict = {}

    for key, val in init_dict.items():
        dko_index_mapping_dict[val] = key

    return dko_index_mapping_dict


def build_tko_index_mapping_dict(sko_index_mapping_dict):

    count = 0

    init_dict = {}

    for key_1, val_1 in sko_index_mapping_dict.items():

        for key_2, val_2 in sko_index_mapping_dict.items():

            for key_3, val_3 in sko_index_mapping_dict.items():

                if key_1 == key_2:
                    continue

                if key_1 == key_3:
                    continue

                if key_2 == key_3:
                    continue

                tko_str_1 = f"{val_1}-{val_2}-{val_3}"
                tko_str_2 = f"{val_1}-{val_3}-{val_2}"
                tko_str_3 = f"{val_2}-{val_1}-{val_3}"
                tko_str_4 = f"{val_2}-{val_3}-{val_1}"
                tko_str_5 = f"{val_3}-{val_1}-{val_2}"
                tko_str_6 = f"{val_3}-{val_2}-{val_1}"

                if (tko_str_1 not in init_dict) and (tko_str_2 not in init_dict) and (tko_str_3 not in init_dict) and (tko_str_4 not in init_dict) and (tko_str_5 not in init_dict) and (tko_str_6 not in init_dict):

                    init_dict[tko_str_1] = count
                    count += 1

    tko_index_mapping_dict = {}

    for key, val in init_dict.items():
        tko_index_mapping_dict[val] = key

    return tko_index_mapping_dict


def get_params_dict(params):

    params_dict = {}

    for name, value in params:

        _, param_type, param_name = name.split('.')

        if param_name not in params_dict:
            params_dict[param_name] = {}

        params_dict[param_name][param_type] = value

    return params_dict


def get_effects_df(params_dict,
                   sko_index_mapping_dict,
                   n_guides):

    dko_index_mapping_dict = build_dko_index_mapping_dict(sko_index_mapping_dict)
    tko_index_mapping_dict = build_tko_index_mapping_dict(sko_index_mapping_dict)

    effects_df = pd.DataFrame(columns=['index', 'ko_type', 'g1', 'g2', 'g3', 'loc', 'scale'])

    for param_name in params_dict:

        match param_name:

            case 'x_i':

                df = pd.DataFrame([range(0, 32),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [sko_index_mapping_dict[i] for i in df['index']]
                df['g2'] = ''
                df['g3'] = ''
                df['ko_type'] = 'guide_sko'

                effects_df = pd.concat([effects_df, df])

            case 'x_ij':

                df = pd.DataFrame([range(0, 496),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [dko_index_mapping_dict[i].split('-')[0] for i in df['index']]
                df['g2'] = [dko_index_mapping_dict[i].split('-')[1] for i in df['index']]
                df['g3'] = ''
                df['ko_type'] = 'guide_dko'

                effects_df = pd.concat([effects_df, df])

            case 'x_ijk':

                df = pd.DataFrame([range(0, 4960),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [tko_index_mapping_dict[i].split('-')[0] for i in df['index']]
                df['g2'] = [tko_index_mapping_dict[i].split('-')[1] for i in df['index']]
                df['g3'] = [tko_index_mapping_dict[i].split('-')[2] for i in df['index']]
                df['ko_type'] = 'guide_tko'

                effects_df = pd.concat([effects_df, df])

            case 'g_u':

                df = pd.DataFrame([range(0, 16),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [sko_index_mapping_dict[(i + 1) * 2 - 1].split('_')[0] for i in df['index']]
                df['g2'] = ''
                df['g3'] = ''
                df['ko_type'] = 'gene_sko'

                effects_df = pd.concat([effects_df, df])

            case 'g_uv':

                df = pd.DataFrame([range(0, 256),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [dko_index_mapping_dict[i].split('-')[0].split('_')[0] for i in df['index']]
                df['g2'] = [dko_index_mapping_dict[i].split('-')[1].split('_')[0] for i in df['index']]
                df['g3'] = ''
                df['ko_type'] = 'gene_dko'

                effects_df = pd.concat([effects_df, df])

            case 'g_uvw':

                df = pd.DataFrame([range(0, 4096),
                                   params_dict[param_name]['locs'].detach().numpy(),
                                   params_dict[param_name]['scales'].detach().numpy()]).T.sort_values(by=1, axis=0, ascending=True)
                df.columns = ['index', 'loc', 'scale']
                df['g1'] = [tko_index_mapping_dict[i].split('-')[0].split('_')[0] for i in df['index']]
                df['g2'] = [tko_index_mapping_dict[i].split('-')[1].split('_')[0] for i in df['index']]
                df['g3'] = [tko_index_mapping_dict[i].split('-')[2].split('_')[0] for i in df['index']]
                df['ko_type'] = 'gene_tko'

                effects_df = pd.concat([effects_df, df])

    return effects_df


def graph_variational_distributions(effects_df,
                                    model_name,
                                    images_path,
                                    x_axis_range=(-3, 3)):

    x_axis = np.arange(x_axis_range[0], x_axis_range[1], 0.001)

    for ko_type in effects_df['ko_type'].unique():

        for _, row in effects_df[effects_df['ko_type'] == ko_type].iterrows():
            plt.plot(x_axis, norm.pdf(x_axis, row['loc'], row['scale']))

        plt.title(f"{model_name}: {ko_type} inferred variational distributions")
        plt.xlabel('x')
        plt.ylabel('density')

        plt.savefig(f"{images_path}/{model_name}_{ko_type}_variational_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()


def graph_inferred_mus(effects_df, model_name, images_path, y_axis_range=(-1.5, 1.5)):

    for ko_type in effects_df['ko_type'].unique():

        locs_df = effects_df[effects_df['ko_type'] == ko_type][['g1', 'g2', 'g3', 'loc']]

        plt.scatter(range(0, locs_df.shape[0]), locs_df['loc'])
        plt.ylabel(f"inferred {ko_type} mu")
        plt.title(f"inferred {ko_type} effect")

        if locs_df.shape[0] < 50:
            plt.xticks(range(0, locs_df.shape[0]), locs_df[['g1', 'g2', 'g3']].agg('-'.join, axis=1), rotation='vertical')

        plt.axhline(y=0.0, color='k', linestyle=':')
        plt.ylim(y_axis_range[0], y_axis_range[1])

        plt.savefig(f"{images_path}/{model_name}_{ko_type}_mu.png", dpi=300, bbox_inches='tight')
        plt.show()


def graph_inferred_mus_and_variational_distributions(effects_df,
                                                     model_name,
                                                     images_path,
                                                     mus_y_axis_range=(-1.5, 1.5),
                                                     var_dist_x_axis_range=(-3, 3)):

    for ko_type in effects_df['ko_type'].unique():

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        locs_df = effects_df[effects_df['ko_type'] == ko_type][['g1', 'g2', 'g3', 'loc']]

        ax[0].scatter(range(0, locs_df.shape[0]), locs_df['loc'])
        ax[0].set_ylabel(f"inferred {ko_type} mu")
        ax[0].set_title(f"inferred {ko_type} effect")

        if locs_df.shape[0] < 50:
            ax[0].set_xticks(range(0, locs_df.shape[0]), locs_df[['g1', 'g2', 'g3']].agg('-'.join, axis=1), rotation='vertical')

        ax[0].axhline(y=0.0, color='k', linestyle=':')
        ax[0].set_ylim(mus_y_axis_range[0], mus_y_axis_range[1])

        x_axis = np.arange(var_dist_x_axis_range[0], var_dist_x_axis_range[1], 0.001)

        for _, row in effects_df[effects_df['ko_type'] == ko_type].iterrows():
            ax[1].plot(x_axis, norm.pdf(x_axis, row['loc'], row['scale']))

        ax[1].set_title(f"{model_name}: {ko_type} inferred variational distributions")
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('density')

        plt.savefig(f"{images_path}/{model_name}_{ko_type}_mus_and_variational_distributions.png", dpi=300, bbox_inches='tight')

        plt.show()


def graph_inferred_taus(params_dict, model_name, images_path):

    plt.hist(params_dict['tau_s']['locs'].detach().numpy(), bins=100)
    plt.xlabel('inferred tau_s')
    plt.ylabel('count')
    plt.title('inferred sample variance')

    plt.savefig(f"{images_path}/{model_name}_tau_s.png", dpi=300, bbox_inches='tight')
    plt.show()
