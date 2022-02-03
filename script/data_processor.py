#! /home/rdaneel/anaconda3/lib/python3.8
# -*- coding: UTF-8 -*-
"""Data processor"""

import os
import argparse
from tkinter import W
from typing import Dict, List, Tuple
import yaml
import matplotlib.pyplot as plt
import util
import numpy as np

class DataProcessor:
    def __init__(self, in_config) -> None:
        self.config: Dict = dict()
        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), in_config)
        with open(config_dir, 'r') as fin:
            self.config = yaml.load(fin, Loader=yaml.FullLoader)

        # Plot parameters
        self.fig_size:Tuple[int,int] = (12, 8)
        self.marker_size:int = 15
        self.line_width:float = 1.8
        self.text_size:int = 20

    def get_num_val(self, in_index:str='succ'):
        """Compute the success rate versus the numbers of agents

        Args:
            in_index (str, optional): which data we want to analyze. Defaults to 'succ'.

        Returns:
            Dict: the success rate (versus the numbers of agents) of each solver
        """

        result: Dict = dict()
        for solver in self.config['solvers']:
            result[solver['name']] = dict()
            for _map_ in self.config['maps']:
                result[solver['name']][_map_['name']] = {'num': list(), 'val': list(), 'ci': list()}

                for ag_num in _map_['num_of_agents']:
                    total_val = 0.0
                    total_num = 0.0
                    _data_:List = list()
                    for scen in _map_['scens']:
                        data_frame = util.get_csv_instance(self.config['exp_path'], _map_['name'],
                                                scen, ag_num, solver)
                        for _, row in data_frame.iterrows():
                            if (in_index == 'succ'
                                and row['solution cost'] >= 0
                                and row['runtime'] <= self.config['time_limit']):
                                total_val += 1

                            elif in_index != 'succ':
                                if in_index == 'runtime':
                                    _data_.append(min(row[in_index], 60))
                                    total_val += min(row[in_index], 60)
                                    
                                elif in_index == 'flex':
                                    if row['solution cost'] >= 0 and row['runtime'] <= self.config['time_limit']:
                                        # _data_.append(tmp_fw*row['min f value'] - row['solution cost'])
                                        # total_val += tmp_fw*row['min f value'] - row['solution cost']
                                        _data_.append(row['solution cost'])
                                        total_val += row['solution cost']
                                        total_num += 1

                                else:
                                    _data_.append(row[in_index])
                                    total_val += row[in_index]

                        # total_num += self.config['ins_num']
                    if total_num == 0:
                        _rate_ = 0
                    else:
                        _rate_ = total_val / total_num  # average value
                    result[solver['name']][_map_['name']]['num'].append(ag_num)
                    result[solver['name']][_map_['name']]['val'].append(_rate_)

                    # if len(_data_) > 0:  # non empty list
                    #     _ci_ = 1.96*np.std(_data_) / np.sqrt(total_num)  # confident interval
                    #     result[solver['name']][_map_['name']]['ci'].append(_ci_)

        return result

    def get_w_val(self, in_index:str, f_weights:List):
        """Compute the success rate versus the numbers of agents

        Args:
            in_index (str, optional): which data we want to analyze.
            f_weights (List[int], optional): focal weights for x-axis.

        Returns:
            Dict: the success rate (versus the numbers of agents) of each solver
        """

        result: Dict = dict()
        for solver in self.config['solvers']:
            result[solver['name']] = dict()
            for _map_ in self.config['maps']:
                result[solver['name']][_map_['name']] = {'w': list(), 'val': list(), 'ci': list()}
                default_w = solver['w']

                for tmp_fw in f_weights:
                    solver['w'] = tmp_fw
                    total_val = 0.0
                    total_num = 0.0
                    _data_:List = list()

                    for ag_num in _map_['num_of_agents']:
                        for scen in _map_['scens']:
                            data_frame = util.get_csv_instance(self.config['exp_path'],
                                                    _map_['name'], scen, ag_num, solver)
                            for _, row in data_frame.iterrows():
                                if in_index == 'succ':
                                    if row['solution cost'] >= 0 and \
                                        row['runtime'] <= self.config['time_limit']:
                                        total_val += 1

                                elif in_index == 'runtime':
                                    _data_.append(min(row[in_index], 60))
                                    total_val += min(row[in_index], 60)

                                elif in_index == 'flex':
                                    if row['solution cost'] >= 0 and row['runtime'] <= self.config['time_limit']:
                                        # _data_.append(tmp_fw*row['min f value'] - row['solution cost'])
                                        # total_val += tmp_fw*row['min f value'] - row['solution cost']
                                        _data_.append(row['solution cost'])
                                        total_val += row['solution cost']
                                        # total_num += 1
                                
                                elif in_index == 'lb':
                                    _data_.append(row['min f value'])
                                    total_val += row['min f value']

                                else:
                                    _data_.append(row[in_index])
                                    total_val += row[in_index]

                            total_num += self.config['ins_num']

                    _rate_ = total_val / total_num  # average value
                    result[solver['name']][_map_['name']]['w'].append(tmp_fw)
                    result[solver['name']][_map_['name']]['val'].append(_rate_)

                    # if len(_data_) > 0:  # non empty list
                    #     _ci_ = 1.96*np.std(_data_) / np.sqrt(total_num)  # confident interval
                    #     result[solver['name']][_map_['name']]['ci'].append(_ci_)

                solver['w'] = default_w

        return result

    def plot_num_val(self, in_index:str='succ'):
        """Plot the success rate versus the number of agents
        """
        fig_align = (1, len(self.config['maps'])) if len(self.config['maps']) > 1 else (1,1)
        result = self.get_num_val(in_index)
        fig, axs = plt.subplots(nrows=fig_align[0],
                                ncols=fig_align[1],
                                figsize=(15,8),
                                dpi=80,
                                facecolor='w',
                                edgecolor='k')

        for idx, _map_ in enumerate(self.config['maps']):
            frow = idx // (len(self.config['maps']) // 2)
            fcol = idx % (len(self.config['maps']) // 2)
            for solver in self.config['solvers']:
                _num_ = result[solver['name']][_map_['name']]['num']
                _val_ = result[solver['name']][_map_['name']]['val']
                _ci_  = result[solver['name']][_map_['name']]['ci']

                if len(_ci_) > 0:
                    _lb_ = [_val_[i] - _ci_[i] for i in range(len(_val_))]
                    _ub_ = [_val_[i] + _ci_[i] for i in range(len(_val_))]
                    axs[frow].fill_between(_num_, _lb_, _ub_,
                                                 color=solver['color'], alpha=0.2)

                if frow == 0 and fcol == 0:
                    axs[frow].plot(_num_, _val_,
                                         label=solver['label'],
                                         linewidth=self.line_width,
                                         markerfacecolor='white',
                                         markeredgewidth=self.line_width,
                                         ms=self.marker_size,
                                         color=solver['color'],
                                         marker=solver['marker'])
                else:
                    axs[frow].plot(_num_, _val_,
                                         linewidth=self.line_width,
                                         markerfacecolor='white',
                                         markeredgewidth=self.line_width,
                                         ms=self.marker_size,
                                         color=solver['color'],
                                         marker=solver['marker'])

                axs[frow].set_title(_map_['label'], fontsize=self.text_size)
                axs[frow].set_xlabel('Number of agents', fontsize=self.text_size)
                axs[frow].axes.set_xticks(_num_)
                axs[frow].axes.set_xticklabels(_num_, fontsize=self.text_size)

                if in_index == 'succ':
                    y_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    axs[frow].set_ylabel('Success rate', fontsize=self.text_size)
                    axs[frow].axes.set_yticks(y_list)
                    axs[frow].axes.set_yticklabels(y_list, fontsize=self.text_size)

                elif in_index == 'runtime':
                    y_list = range(0, 61, 10)
                    axs[frow].set_ylabel('Runtime (sec)', fontsize=self.text_size)
                    axs[frow].axes.set_yticks(y_list)
                    axs[frow].axes.set_yticklabels(y_list, fontsize=self.text_size)

                elif in_index == 'flex':
                    axs[frow].set_ylabel('Cost', fontsize=self.text_size)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.legend(loc="upper center",
                   bbox_to_anchor= (0.5, 1.01),
                   borderpad=0.25, handletextpad=0.1, labelspacing=0.75, columnspacing=0.75,
                   ncol=len(self.config['solvers'])//2,
                   fontsize=self.text_size)
        plt.savefig('./tmp.png')
        plt.show()


    def plot_w_val(self, in_index:str, f_weights:List):
        """Plot the success rate versus the focal weights
        """
        fig_align = (1, len(self.config['maps'])) if len(self.config['maps']) > 1 else (1,1)
        result = self.get_w_val(in_index, f_weights)
        fig, axs = plt.subplots(nrows=fig_align[0],
                                ncols=fig_align[1],
                                figsize=(15,8),
                                dpi=80,
                                facecolor='w',
                                edgecolor='k')

        for idx, _map_ in enumerate(self.config['maps']):
            frow = idx // (len(self.config['maps']) // 2)
            fcol = idx % (len(self.config['maps']) // 2)
            for solver in self.config['solvers']:
                _num_ =  range(0, len(f_weights), 1)
                _val_ = result[solver['name']][_map_['name']]['val']
                _ci_  = result[solver['name']][_map_['name']]['ci']

                if len(_ci_) > 0:
                    _lb_ = [_val_[i] - _ci_[i] for i in range(len(_val_))]
                    _ub_ = [_val_[i] + _ci_[i] for i in range(len(_val_))]
                    axs[frow, fcol].fill_between(_num_, _lb_, _ub_,
                                                 color=solver['color'], alpha=0.2)

                if len(self.config['maps'])//2 == 1:
                    if frow == 0 and fcol == 0:
                        axs[frow].plot(_num_, _val_,
                            label=solver['label'],
                            linewidth=self.line_width,
                            markerfacecolor='white',
                            markeredgewidth=self.line_width,
                            ms=self.marker_size,
                            color=solver['color'],
                            marker=solver['marker'])
                    else:
                        axs[frow].plot(_num_, _val_,
                            linewidth=self.line_width,
                            markerfacecolor='white',
                            markeredgewidth=self.line_width,
                            ms=self.marker_size,
                            color=solver['color'],
                            marker=solver['marker'])
                else:
                    axs[frow, fcol].plot(_num_, _val_,
                                    label=solver['label'],
                                    linewidth=self.line_width,
                                    markerfacecolor='white',
                                    markeredgewidth=self.line_width,
                                    ms=self.marker_size,
                                    color=solver['color'],
                                    marker=solver['marker'])

                # if frow == 0 and fcol == 0:
                #     if len(self.config['maps'])//2 == 1:
                #         axs[frow].plot(_num_, _val_,
                #                         label=solver['label'],
                #                         linewidth=self.line_width,
                #                         markerfacecolor='white',
                #                         markeredgewidth=self.line_width,
                #                         ms=self.marker_size,
                #                         color=solver['color'],
                #                         marker=solver['marker'])
                #     else:
                #         axs[frow, fcol].plot(_num_, _val_,
                #                          label=solver['label'],
                #                          linewidth=self.line_width,
                #                          markerfacecolor='white',
                #                          markeredgewidth=self.line_width,
                #                          ms=self.marker_size,
                #                          color=solver['color'],
                #                          marker=solver['marker'])
                # else:
                #     if len(self.config['maps'])//2 == 1:
                #         axs[frow].plot(_num_, _val_,
                #                         label=solver['label'],
                #                         linewidth=self.line_width,
                #                         markerfacecolor='white',
                #                         markeredgewidth=self.line_width,
                #                         ms=self.marker_size,
                #                         color=solver['color'],
                #                         marker=solver['marker'])
                #     else:
                #         axs[frow, fcol].plot(_num_, _val_,
                #                          linewidth=self.line_width,
                #                          markerfacecolor='white',
                #                          markeredgewidth=self.line_width,
                #                          ms=self.marker_size,
                #                          color=solver['color'],
                #                          marker=solver['marker'])

                if len(self.config['maps'])//2 == 1:
                    axs[frow].set_title(_map_['label'], fontsize=self.text_size)
                    axs[frow].set_xlabel('Suboptimality factor', fontsize=self.text_size)
                    axs[frow].axes.set_xticks(_num_)
                    axs[frow].axes.set_xticklabels(f_weights, fontsize=self.text_size)

                    if in_index == 'succ':
                        y_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        axs[frow].set_ylabel('Success rate', fontsize=self.text_size)
                        axs[frow].axes.set_yticks(y_list)
                        axs[frow].axes.set_yticklabels(y_list, fontsize=self.text_size)

                    elif in_index == 'runtime':
                        y_list = range(0, 61, 10)
                        axs[frow].set_ylabel('Runtime (sec)', fontsize=self.text_size)
                        axs[frow].axes.set_yticks(y_list)
                        axs[frow].axes.set_yticklabels(y_list, fontsize=self.text_size)
                        axs[frow].set_ylim(-5+min(y_list), max(y_list)+5)
                    
                    elif in_index == 'flex':
                        axs[frow].set_ylabel('Cost', fontsize=self.text_size)
                        
                    elif in_index == 'lb':
                        axs[frow].set_ylabel('Lower bound', fontsize=self.text_size)
                    
                else:
                    axs[frow, fcol].set_title(_map_['label'], fontsize=self.text_size)
                    axs[frow, fcol].set_xlabel('Suboptimality factor', fontsize=self.text_size)
                    axs[frow, fcol].axes.set_xticks(_num_)
                    axs[frow, fcol].axes.set_xticklabels(f_weights, fontsize=self.text_size)

                    if in_index == 'succ':
                        y_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        axs[frow, fcol].set_ylabel('Success rate', fontsize=self.text_size)
                        axs[frow, fcol].axes.set_yticks(y_list)
                        axs[frow, fcol].axes.set_yticklabels(y_list, fontsize=self.text_size)

                    elif in_index == 'runtime':
                        y_list = range(0, 61, 10)
                        axs[frow, fcol].set_ylabel('Runtime (sec)', fontsize=self.text_size)
                        axs[frow, fcol].axes.set_yticks(y_list)
                        axs[frow, fcol].axes.set_yticklabels(y_list, fontsize=self.text_size)
                        axs[frow, fcol].set_ylim(-5+min(y_list), max(y_list)+5)
                    
                    elif in_index == 'flex':
                        axs[frow, fcol].set_ylabel('Cost', fontsize=self.text_size)
                        
                    elif in_index == 'lb':
                        axs[frow, fcol].set_ylabel('Lower bound', fontsize=self.text_size)
                        axs[frow, fcol].axes.set_yticklabels(fontsize=self.text_size)
                        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.legend(loc="upper center",
                   bbox_to_anchor= (0.5, 1.01),
                   borderpad=0.25, handletextpad=0.1, labelspacing=0.75, columnspacing=0.75,
                   ncol=len(self.config['solvers']),
                   fontsize=self.text_size)
        plt.savefig('./tmp2.png')
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()

    # Create data processor
    date_processor = DataProcessor(args.config)
    # date_processor.plot_num_val('runtime')
    date_processor.plot_num_val('succ')
    # date_processor.plot_num_val('flex')
    # date_processor.plot_w_val('succ', [1.01, 1.02, 1.05, 1.10])
    # date_processor.plot_w_val('lb', [1.01, 1.02, 1.05, 1.10])
    # date_processor.plot_w_val('flex', [1.01, 1.02, 1.05, 1.10])
