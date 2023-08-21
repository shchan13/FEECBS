#pragma once
#include "ECBSNode.h"

class IterTracker
{
public:
    int num_of_agents;
    double suboptimality;

    vector<int> iter_sum_lb;
    vector<int> br_sum_lb;
    vector<int> all_sum_lb;
    vector<int> open_sum_lb;

	vector<int> iter_sum_fval;
    vector<int> br_sum_fval;
    vector<int> all_sum_fval;
    vector<int> open_sum_fval;

    vector<int> iter_sum_cost;
    vector<int> br_sum_cost;
    vector<int> all_sum_cost;
    vector<int> open_sum_cost;
    
    vector<int> iter_num_conflicts;
    vector<int> br_num_conflicts;
    vector<int> all_num_conflicts;
    vector<int> open_num_conflicts;

    vector<double> iter_remained_flex;
    vector<double> br_remained_flex;
    vector<double> all_remained_flex;
    vector<double> open_remained_flex;

	vector<double> iter_subopt;
    vector<double> br_subopt;
    vector<double> all_subopt;

    vector<uint64_t> iter_sum_ll_generate;
    vector<uint64_t> br_sum_ll_generate;
    vector<uint64_t> all_sum_ll_generate;
	vector<uint64_t> replan_ll_generate;
	vector<int> replan_agent;
	vector<double> replan_flex;

    vector<int> iter_node_idx;
    vector<int> br_node_idx;
    vector<int> open_node_idx;
    vector<int> all_node_idx;

	vector<uint64_t> iter_num_focal;
	vector<uint64_t> iter_num_open;
	vector<uint64_t> iter_num_cleanup;
	vector<int> iter_node_type;

	vector<bool> iter_use_flex;
	vector<bool> iter_no_more_flex;
	vector<bool> iter_cannot_use_flex;

	vector<vector<int>> iter_ag_lb;
	vector<vector<int>> br_ag_lb;

	vector<vector<int>> iter_ag_cost;
	vector<vector<int>> br_ag_cost;

    IterTracker() {};
    void getCleanupHeadStats(ECBSNode *cleanup_head,
        size_t focal_num, size_t open_num, size_t cleanup_num);
    void getIterStats(HLNode *node, int cleanup_head_lb,
        const vector<int>& min_f_vals, const vector<Path*>& paths);
    void getBranchStats(HLNode *node, int cleanup_head_lb,
        const vector<int>& init_min_f_vals, const vector<int>& init_costs);
    void getAllStats(HLNode *node);
    void saveIterStats(const string& file_path);
    void saveNumNodesInLists(const string& file_path);
};
