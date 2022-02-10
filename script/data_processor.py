#! /home/rdaneel/anaconda3/lib/python3.8
# -*- coding: UTF-8 -*-
"""Data processor"""

import enum
import os
import argparse
from typing import Dict, List, Tuple
import pprint
from sqlalchemy import false, true
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
        self.max_x_num = 5  # on the x-axis
        self.fig_size:Tuple[int,int] = (17, 8)
        self.marker_size:int = 15
        self.line_width:float = 1.8
        self.text_size:int = 25
        self.fig_axs:Dict[int, Tuple[int,int]] = {1: (1,1),
                                                  2: (1,2),
                                                  3: (1,3),
                                                  4: (2,2),
                                                  5: (1,5),
                                                  6: (2,3),
                                                  8: (2,4),
                                                  9: (3,3)}
        self.y_labels:Dict[str, str] = {'succ': 'Success rate',
                                        'runtime': 'Runtime (sec)',
                                        'min f value': 'Lower bound',
                                        'solution cost': 'Cost',
                                        'Ratio': 'Ratio',
                                        '#findPathForSingleAgent': 'Number of v-t nodes (K)',
                                        'flex': '$\Delta$'}
        self.x_labels:Dict[str,str] = {'num': 'Number of agents',
                                       'ins': 'Instance',
                                       'w': 'Suboptimality factor'}


    def get_subfig_pos(self, f_idx: int):
        """Transfer subplot index to 2-D position
        Args:
            f_idx (int): subplot index

        Returns:
            int, int: 2D position
        """
        f_row = self.fig_axs[len(self.config['maps'])][1]
        return f_idx // f_row, f_idx % f_row


    def get_val(self, x_index:str='num', y_index:str='succ', is_avg:bool=true):
        if x_index == 'ins':
            return self.get_ins_val(y_index)
        elif x_index == 'num':
            return self.get_num_val(y_index, is_avg)
        elif x_index == 'w':
            return self.get_w_val(y_index, is_avg)


    def get_ins_val(self, in_index:str='runtime'):
        """Compute the success rate versus the numbers of agents

        Args:
            in_index (str, optional): which data we want to analyze. Defaults to 'runtime'.

        Returns:
            Dict: the success rate (versus the numbers of agents) of each solver
        """

        result: Dict = dict()
        for solver in self.config['solvers']:
            result[solver['name']] = dict()
            for _map_ in self.config['maps']:
                result[solver['name']][_map_['name']] = {'x': list(), 'val': list(), 'ci': list()}
                global_idx = 1

                for ag_num in _map_['num_of_agents']:
                    for scen in _map_['scens']:
                        data_frame = util.get_csv_instance(self.config['exp_path'], _map_['name'],
                                                           scen, ag_num, solver)
                        for _, row in data_frame.iterrows():
                            if in_index == 'runtime':
                                tmp_val = min(row[in_index], 60)
                            elif in_index == 'temp':
                                tmp_val = row['#low-level in focal']/row['#findPathForSingleAgent']
                            elif row[in_index] < 0:
                                tmp_val = np.inf
                            else:
                                tmp_val = row[in_index]

                            result[solver['name']][_map_['name']]['val'].append(tmp_val)
                            result[solver['name']][_map_['name']]['x'].append(global_idx)
                            global_idx += 1
        return result


    def get_num_val(self, in_index:str='succ', is_avg:bool=true):
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
                result[solver['name']][_map_['name']] = {'x': list(), 'val': list(), 'ci': list()}

                for ag_num in _map_['num_of_agents']:
                    total_val = 0.0
                    total_num = 0.0
                    _data_:List = list()
                    for scen in _map_['scens']:
                        data_frame = util.get_csv_instance(self.config['exp_path'], _map_['name'],
                                                           scen, ag_num, solver)
                        for _, row in data_frame.iterrows():
                            if in_index == 'succ':
                                if row['solution cost'] >= 0 and \
                                    row['runtime'] <= self.config['time_limit']:
                                    total_val += 1
                            elif in_index == 'runtime':
                                _data_.append(min(row[in_index], 60))
                                total_val += min(row[in_index], 60)
                            else:
                                assert row[in_index] >= 0
                                _data_.append(row[in_index])
                                total_val += row[in_index]

                        total_num += self.config['ins_num']

                    if is_avg:
                        if total_num == 0:
                            _rate_ = 0
                        else:
                            _rate_ = total_val / total_num  # average value
                    else:
                        _rate_ = total_val

                    result[solver['name']][_map_['name']]['x'].append(ag_num)
                    result[solver['name']][_map_['name']]['val'].append(_rate_)

                    if self.config['plot_ci'] and len(_data_) > 0:  # non empty list
                        _ci_ = 1.96*np.std(_data_) / np.sqrt(total_num)  # confident interval
                        result[solver['name']][_map_['name']]['ci'].append(_ci_)

        return result


    def get_w_val(self, in_index:str, is_avg:bool=true):
        """Compute the success rate versus the numbers of agents

        Args:
            in_index (str, optional): which data we want to analyze.

        Returns:
            Dict: the success rate (versus the numbers of agents) of each solver
        """

        result: Dict = dict()
        for solver in self.config['solvers']:
            result[solver['name']] = dict()
            for _map_ in self.config['maps']:
                result[solver['name']][_map_['name']] = {'x': list(), 'val': list(), 'ci': list()}
                default_w = solver['w']

                for tmp_fw in self.config['f_weights']:
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
                                else:
                                    assert row[in_index] >= 0
                                    _data_.append(row[in_index])
                                    total_val += row[in_index]

                            total_num += self.config['ins_num']

                    if is_avg:
                        _rate_ = total_val / total_num  # average value
                    else:
                        _rate_ = total_val

                    result[solver['name']][_map_['name']]['x'].append(tmp_fw)
                    result[solver['name']][_map_['name']]['val'].append(_rate_)

                    if self.config['plot_ci'] and len(_data_) > 0:  # non empty list
                        _ci_ = 1.96*np.std(_data_) / np.sqrt(total_num)  # confident interval
                        result[solver['name']][_map_['name']]['ci'].append(_ci_)

                solver['x'] = default_w

        return result


    def subplot_fig(self, x_index, y_index, in_axs, in_map_idx, in_map, in_result):
        _x_ = in_result[self.config['solvers'][0]['name']][in_map['name']]['x']
        _num_ = range(1, len(_x_)+1)

        for solver in self.config['solvers']:
            _val_ = in_result[solver['name']][in_map['name']]['val']
            _ci_  = in_result[solver['name']][in_map['name']]['ci']

            if in_map_idx == 0:
                in_axs.plot(_num_, _val_,
                            label=solver['label'],
                            linewidth=self.line_width,
                            markerfacecolor='white',
                            markeredgewidth=self.line_width,
                            ms=self.marker_size,
                            color=solver['color'],
                            marker=solver['marker'])
            else:
                in_axs.plot(_num_, _val_,
                            linewidth=self.line_width,
                            markerfacecolor='white',
                            markeredgewidth=self.line_width,
                            ms=self.marker_size,
                            color=solver['color'],
                            marker=solver['marker'])

            # Plot confident interval
            if self.config['plot_ci'] and len(_ci_) > 0:
                _lb_ = [_val_[i] - _ci_[i] for i in range(len(_val_))]
                _ub_ = [_val_[i] + _ci_[i] for i in range(len(_val_))]
                in_axs.fill_between(_num_, _lb_, _ub_, color=solver['color'], alpha=0.2)

        in_axs.set_title(in_map['label'], fontsize=self.text_size)

        if len(_num_) > self.max_x_num:
            _num_ = list(range(len(_x_)//self.max_x_num, len(_x_)+1, len(_x_)//self.max_x_num))
            _num_.insert(0, 1)
            _x_ = _num_

        in_axs.axes.set_xticks(_num_)
        in_axs.axes.set_xticklabels(_x_, fontsize=self.text_size)
        in_axs.set_xlabel(self.x_labels[x_index], fontsize=self.text_size)

        y_list = in_axs.axes.get_yticks()
        if y_index == 'succ':
            y_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            in_axs.axes.set_yticks(y_list)
        elif y_index == 'runtime':
            y_list = range(0, 61, 10)
            in_axs.axes.set_yticks(y_list)
        elif y_index == '#findPathForSingleAgent':
            y_list = np.arange(min(y_list), max(y_list)+5, 1000)
            y_list = np.delete(y_list, 0)
            in_axs.axes.set_yticks(y_list)
            y_list = [str(int(y//1000)) for y in y_list]

        in_axs.axes.set_yticklabels(y_list, fontsize=self.text_size)
        in_axs.set_ylabel(self.y_labels[y_index], fontsize=self.text_size)

    def subplot_fig2(self, x_index, y_index, in_axs, in_result):
        _x_ = in_result[self.config['solvers'][0]['name']]['x']
        _num_ = range(1, len(_x_)+1)

        for solver in self.config['solvers']:
            _val_ = in_result[solver['name']]['val']

            in_axs.plot(_num_, _val_,
                        label=solver['label'],
                        linewidth=self.line_width,
                        markerfacecolor='white',
                        markeredgewidth=self.line_width,
                        ms=self.marker_size,
                        color=solver['color'],
                        marker=solver['marker'])

        if len(_num_) > self.max_x_num:
            _num_ = list(range(len(_x_)//self.max_x_num, len(_x_)+1, len(_x_)//self.max_x_num))
            _num_.insert(0, 1)
            _x_ = _num_

        in_axs.axes.set_xticks(_num_)
        in_axs.axes.set_xticklabels(_x_, fontsize=self.text_size)
        in_axs.set_xlabel(self.x_labels[x_index], fontsize=self.text_size)

        y_list = in_axs.axes.get_yticks()
        if y_index == 'succ':
            y_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            in_axs.axes.set_yticks(y_list)
        elif y_index == 'runtime':
            y_list = range(0, 61, 10)
            in_axs.axes.set_yticks(y_list)
        in_axs.axes.set_yticklabels(y_list, fontsize=self.text_size)
        in_axs.set_ylabel(self.y_labels[y_index], fontsize=self.text_size)


    def get_sum_along_w(self, y_index:str='succ'):
        result = self.get_val('w', y_index)
        tmp_sum = dict()
        for tmp_solver in self.config['solvers']:
            tmp_sum[tmp_solver['name']] = dict()
            for tmp_w_idx, tmp_w in enumerate(self.config['f_weights']):
                tmp_tmp_sum = 0
                for tmp_map in self.config['maps']:
                    tmp_tmp_sum += result[tmp_solver['name']][tmp_map['name']]['val'][tmp_w_idx]
                tmp_sum[tmp_solver['name']][tmp_w] = tmp_tmp_sum
        pprint.pprint(tmp_sum)
        return tmp_sum

    def get_lb_improvement_along_w(self):
        result1 = self.get_val('w', 'min f value')
        result2 = self.get_val('w', ' root f value')
        tmp_sum = dict()
        for tmp_solver in self.config['solvers']:
            tmp_sum[tmp_solver['name']] = dict()
            for tmp_w_idx, tmp_w in enumerate(self.config['f_weights']):
                tmp_tmp_sum = 0
                for tmp_map in self.config['maps']:
                    tmp_val = result1[tmp_solver['name']][tmp_map['name']]['val'][tmp_w_idx] - result2[tmp_solver['name']][tmp_map['name']]['val'][tmp_w_idx]
                    tmp_tmp_sum += tmp_val
                tmp_sum[tmp_solver['name']][tmp_w] = tmp_tmp_sum
        pprint.pprint(tmp_sum)
        return tmp_sum

    def plot_fig(self, x_index:str='num', y_index:str='succ'):
        # Get the result from the experiments
        result = self.get_val(x_index, y_index)

        # Plot all the subplots on the figure
        fig, axs = plt.subplots(nrows=self.fig_axs[len(self.config['maps'])][0],
                                ncols=self.fig_axs[len(self.config['maps'])][1],
                                figsize=self.fig_size,
                                dpi=80, facecolor='w', edgecolor='k')

        for idx, _map_ in enumerate(self.config['maps']):
            frow, fcol = self.get_subfig_pos(idx)
            if len(self.config['maps']) == 1:
                self.subplot_fig(x_index, y_index, axs, idx, _map_, result)
            elif self.fig_axs[len(self.config['maps'])][0] == 1:
                self.subplot_fig(x_index, y_index, axs[fcol], idx, _map_, result)
            else:
                self.subplot_fig(x_index, y_index, axs[frow,fcol], idx, _map_, result)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.legend(loc="upper center",
                   bbox_to_anchor= (0.5, 1.01),
                   borderpad=0.25, handletextpad=0.1, labelspacing=0.75, columnspacing=0.75,
                   ncol=len(self.config['solvers']),
                   fontsize=self.text_size)
        fig_name = x_index + '_' + y_index + '_plot.png'
        plt.savefig(fig_name)
        plt.show()

    def plot_fig2(self, x_index:str='num', y_index:str='succ'):
        # Get the result from the experiments
        y = '#findPathForSingleAgent'
        sum_ag_plan = self.get_val(x_index, y, false)
        ll_node_focal = self.get_val(x_index, '#low-level in focal', false)

        result = dict()
        for tmp_solver in self.config['solvers']:
            result[tmp_solver['name']] = {'x':list(), 'val':list()}
            for tmp_w_idx, tmp_w in enumerate(self.config['f_weights']):
                tmp_tmp_sum = 0
                for tmp_map in self.config['maps']:
                    tmp_val = ll_node_focal[tmp_solver['name']][tmp_map['name']]['val'][tmp_w_idx] / sum_ag_plan[tmp_solver['name']][tmp_map['name']]['val'][tmp_w_idx]
                    tmp_tmp_sum += tmp_val
                result[tmp_solver['name']]['x'].append(tmp_w)
                result[tmp_solver['name']]['val'].append(tmp_tmp_sum)

        # Plot all the subplots on the figure
        fig, axs = plt.subplots(nrows=1,
                                ncols=1,
                                figsize=self.fig_size,
                                dpi=80, facecolor='w', edgecolor='k')

        self.subplot_fig2(x_index, 'Ratio', axs, result)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.legend(loc="upper center",
                   bbox_to_anchor= (0.5, 1.01),
                   borderpad=0.25, handletextpad=0.1, labelspacing=0.75, columnspacing=0.75,
                   ncol=len(self.config['solvers']),
                   fontsize=self.text_size)
        fig_name = x_index + '_' + y_index + '_plot.png'
        plt.savefig(fig_name)
        plt.show()

    def plot_fig3(self, x_index:str='w', y_index:str='#findPathForSingleAgent'):
        # Get the result from the experiments
        sum_ag_plan = self.get_val(x_index, y_index, false)
        ll_node_focal = self.get_val(x_index, '#low-level in focal', false)

        result = dict()
        for solver in self.config['solvers']:
            result[solver['name']] = dict()
            for _map_ in self.config['maps']:
                result[solver['name']][_map_['name']] = {'x': list(), 'val': list(), 'ci': list()}
                for tmp_w_idx, tmp_fw in enumerate(self.config['f_weights']):
                    tmp_val = ll_node_focal[solver['name']][_map_['name']]['val'][tmp_w_idx] / sum_ag_plan[solver['name']][_map_['name']]['val'][tmp_w_idx]
                    result[solver['name']][_map_['name']]['x'].append(tmp_fw)
                    result[solver['name']][_map_['name']]['val'].append(tmp_val)

        # Plot all the subplots on the figure
        fig, axs = plt.subplots(nrows=self.fig_axs[len(self.config['maps'])][0],
                                ncols=self.fig_axs[len(self.config['maps'])][1],
                                figsize=self.fig_size,
                                dpi=80, facecolor='w', edgecolor='k')

        for idx, _map_ in enumerate(self.config['maps']):
            frow, fcol = self.get_subfig_pos(idx)
            if len(self.config['maps']) == 1:
                self.subplot_fig(x_index, y_index, axs, idx, _map_, result)
            elif self.fig_axs[len(self.config['maps'])][0] == 1:
                self.subplot_fig(x_index, y_index, axs[fcol], idx, _map_, result)
            else:
                self.subplot_fig(x_index, y_index, axs[frow,fcol], idx, _map_, result)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.legend(loc="upper center",
                   bbox_to_anchor= (0.5, 1.01),
                   borderpad=0.25, handletextpad=0.1, labelspacing=0.75, columnspacing=0.75,
                   ncol=len(self.config['solvers']),
                   fontsize=self.text_size)
        fig_name = x_index + '_' + y_index + '_plot.png'
        plt.savefig(fig_name)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()

    # Create data processor
    data_processor = DataProcessor(args.config)
    # data_processor.plot_fig()
    data_processor.plot_fig(x_index='num', y_index='succ')
    # data_processor.plot_fig(x_index='w', y_index='runtime')
    # data_processor.get_sum_along_w()
    # data_processor.get_lb_improvement_along_w()
    # data_processor.plot_fig(x_index='w', y_index='min f value')
    # data_processor.plot_fig(x_index='ins', y_index='temp')
    # data_processor.plot_fig(x_index='w', y_index='runtime')
    # data_processor.plot_fig2(x_index='w')
    # data_processor.plot_fig(x_index='w', y_index='succ')
    # data_processor.plot_fig3()
