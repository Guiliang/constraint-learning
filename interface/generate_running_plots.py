import os

from utils.data_utils import read_running_logs_by_cid, compute_moving_average, mean_std_plot_results
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
                 legend_size=10,
                 axis_size=None,
                 img_size=None,
                 title=None):
    plot_mean_y_dict = {}
    plot_std_y_dict = {}
    plot_x_dict = {}
    for method_name in method_names:
        plot_x_dict.update({method_name: [i for i in range(len(mean_results_moving_avg_dict[method_name]))]})
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (5.7, 5.6),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      axis_size=axis_size if axis_size is not None else 18,
                      title_size=20,
                      plot_name='./plot_results/{0}'.format(save_label), )


def generate_plots():
    env_id = 'HCWithPos-v0'
    method_names_labels_dict = {
        # "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-4_ft-2e5": 'MEICRL_cbs-64_lr-1e-4_ft-2e5',
        # "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-4_ft-2e5": 'MEICRL_cbs-64_lr-3e-4_ft-2e5',
        # "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-4_ft-2e5": 'MEICRL_cbs-64_lr-5e-4_ft-2e5',
        # "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg": 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg',
        # 'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-5_ft-2e5_exp-neg': 'MEICRL_cbs-64_lr-1e-5_ft-2e5_exp-neg',
        # 'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-5_ft-2e5_exp-neg': 'MEICRL_cbs-64_lr-3e-5_ft-2e5_exp-neg',
        # 'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5': 'MEICRL_cbs-64_lr-5e-5_ft-2e5',
        # 'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_additive': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_additive',
        'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1',
        'semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1',
        'semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1',
        'semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1',
        'semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1',
        'semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1_reprob': 'MEICRL_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1_reprob',

    }

    modes = ['train']
    plot_mode = 'all_Sanity'
    img_size = None
    axis_size = None
    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if method_names_labels_dict[method_name] != 'PPO' and method_names_labels_dict[method_name] != 'PPO_lag':
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict
    for mode in modes:
        if env_id == 'HCWithPos-v0':
            max_episodes = 5000
            average_num = 100
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            title = 'Blocked Half-Cheetah'
            log_path_dict = {
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-4_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-4_ft-2e5_exp-neg-multi_env-Oct-28-2022-11:01-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-4_ft-2e5_exp-neg-multi_env-Oct-27-2022-23:16-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-4_ft-2e5_exp-neg-multi_env-Oct-28-2022-11:44-seed_666/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-4_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-4_ft-2e5_exp-neg-multi_env-Oct-28-2022-11:01-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-4_ft-2e5_exp-neg-multi_env-Oct-27-2022-23:16-seed_666/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-4_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-4_ft-2e5_exp-neg-multi_env-Oct-28-2022-11:01-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-4_ft-2e5_exp-neg-multi_env-Oct-27-2022-09:57-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-4_ft-2e5_exp-neg-multi_env-Oct-27-2022-23:16-seed_666/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-multi_env-Oct-28-2022-17:39-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-multi_env-Oct-27-2022-23:16-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-multi_env-Oct-28-2022-12:13-seed_666/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-5_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-1e-5_ft-2e5_exp-neg-multi_env-Oct-31-2022-20:29-seed_123/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-5_ft-2e5_exp-neg": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-3e-5_ft-2e5_exp-neg-multi_env-Oct-31-2022-20:29-seed_123/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5-multi_env-Oct-31-2022-20:33-seed_123/',
                ],
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1": [
                    # '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1-multi_env-Oct-31-2022-20:29-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1-multi_env-Nov-01-2022-11:33-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1-multi_env-Nov-01-2022-11:33-seed_666/',
                ],
                'sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_additive':[
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_additive-multi_env-Nov-01-2022-19:36-seed_123/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1-multi_env-Nov-01-2022-12:03-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1-multi_env-Nov-01-2022-00:02-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1-multi_env-Nov-01-2022-13:20-seed_321/',
                    # '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-1e-1-multi_env-Nov-02-2022-02:19-seed_666/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1-multi_env-Nov-01-2022-00:02-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1-multi_env-Nov-01-2022-13:02-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-3e-1-multi_env-Nov-02-2022-02:06-seed_666/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1-multi_env-Nov-01-2022-00:02-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1-multi_env-Nov-01-2022-12:52-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-5e-1-multi_env-Nov-02-2022-02:38-seed_666/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1_reprob":[
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1_reprob-multi_env-Nov-01-2022-17:25-seed_123/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1-multi_env-Nov-01-2022-00:02-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1-multi_env-Nov-01-2022-13:46-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_weight-8e-1-multi_env-Nov-02-2022-02:41-seed_666/',
                ],

            }
        else:
            raise ValueError("Unknown env id {0}".format(env_id))

        all_mean_dict_by_cid = {}
        all_std_dict_by_cid = {}
        for aid in range(aid_num):
            all_mean_dict_by_cid.update({aid: {}})
            all_std_dict_by_cid.update({aid: {}})
        for method_name in method_names_labels_dict.keys():
            all_results_by_cid = {}
            for aid in range(aid_num):
                all_results_by_cid.update({aid: []})
            for log_path in log_path_dict[method_name]:
                monitor_path_all = []
                if mode == 'train':
                    run_files = os.listdir(log_path)
                    for file in run_files:
                        if 'monitor' in file:
                            monitor_path_all.append(log_path + file)
                else:
                    monitor_path_all.append(log_path + 'test/test.monitor.csv')

                results_by_cid = read_running_logs_by_cid(monitor_path_all=monitor_path_all, read_keys=plot_key,
                                                          max_episodes=(max_episodes + float(
                                                              max_episodes / 5)) * aid_num,
                                                          max_reward=max_reward, min_reward=min_reward, cid_num=aid_num)
                for aid in range(aid_num):
                    all_results_by_cid[aid].append(results_by_cid[aid])

            for aid in range(aid_num):
                mean_dict, std_dict = mean_std_plot_results(all_results_by_cid[aid])
                all_mean_dict_by_cid[aid].update({method_name: {}})
                all_std_dict_by_cid[aid].update({method_name: {}})

                if not os.path.exists(os.path.join('./plot_results/', env_id)):
                    os.mkdir(os.path.join('./plot_results/', env_id))
                if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                    os.mkdir(os.path.join('./plot_results/', env_id, method_name))

                for idx in range(len(plot_key)):
                    mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                         average_num=average_num)
                    std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                        average_num=average_num)
                    print(method_name)
                    if max_episodes:
                        mean_results_moving_average = mean_results_moving_average[:max_episodes]
                        std_results_moving_average = std_results_moving_average[:max_episodes]
                    all_mean_dict_by_cid[aid][method_name].update({plot_key[idx]: mean_results_moving_average})
                    all_std_dict_by_cid[aid][method_name].update({plot_key[idx]: std_results_moving_average / 2})
                    plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                                 std_results_moving_avg_dict={method_name: std_results_moving_average},
                                 label=plot_key[idx],
                                 method_names=[method_name],
                                 ylim=plot_y_lim_dict[plot_key[idx]],
                                 save_label=os.path.join(env_id, method_name,
                                                         plot_key[idx] + '_c{0}_'.format(aid) + '_' + mode),
                                 title=title,
                                 axis_size=axis_size,
                                 img_size=img_size,
                                 )
        for aid in range(aid_num):
            for idx in range(len(plot_key)):
                mean_results_moving_avg_dict = {}
                std_results_moving_avg_dict = {}
                for method_name in method_names_labels_dict.keys():
                    mean_results_moving_avg_dict.update(
                        {method_name: all_mean_dict_by_cid[aid][method_name][plot_key[idx]]})
                    std_results_moving_avg_dict.update(
                        {method_name: all_std_dict_by_cid[aid][method_name][plot_key[idx]]})
                    if (plot_key[idx] == 'reward_nc' or plot_key[idx] == 'constraint') and mode == 'test':
                        print(method_name, plot_key[idx],
                              all_mean_dict_by_cid[aid][method_name][plot_key[idx]][-1],
                              all_std_dict_by_cid[aid][method_name][plot_key[idx]][-1])
                plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                             std_results_moving_avg_dict=std_results_moving_avg_dict,
                             label=label_key[idx],
                             method_names=list(method_names_labels_dict.keys()),
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, plot_key[idx] + '_c{0}_'.format(aid) + '_'
                                                     + mode + '_' + env_id + '_' + plot_mode),
                             # legend_size=18,
                             legend_dict=method_names_labels_dict,
                             title=title,
                             axis_size=axis_size,
                             img_size=img_size,
                             )


if __name__ == "__main__":
    generate_plots()
