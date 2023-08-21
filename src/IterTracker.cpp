#include "IterTracker.h"


void IterTracker::getCleanupHeadStats(ECBSNode *cleanup_head,
	size_t focal_num, size_t open_num, size_t cleanup_num)
{
	open_node_idx.push_back(cleanup_head->time_generated);
	open_sum_lb.push_back(cleanup_head->g_val);
	open_sum_fval.push_back(cleanup_head->getFVal());
	open_sum_cost.push_back(cleanup_head->sum_of_costs);
	open_num_conflicts.push_back(cleanup_head->conflicts.size() + cleanup_head->unknownConf.size());
	open_remained_flex.push_back(suboptimality * cleanup_head->g_val - cleanup_head->sum_of_costs);
	
	iter_num_focal.push_back(focal_num);
	iter_num_open.push_back(open_num);
	iter_num_cleanup.push_back(cleanup_num);
}


void IterTracker::getIterStats(HLNode *node, int cleanup_head_lb,
	const vector<int>& min_f_vals, const vector<Path*>& paths)
{
	int total_num_conf = (int)node->conflicts.size() + (int)node->unknownConf.size();
	double remain_flex = suboptimality * (double)node->g_val - (double)node->sum_of_costs;

	iter_node_idx.push_back(node->time_generated);
	iter_sum_lb.push_back(node->g_val);
	iter_sum_fval.push_back(node->getFVal());
	iter_sum_cost.push_back(node->sum_of_costs);
	iter_num_conflicts.push_back(total_num_conf);
	iter_remained_flex.push_back(remain_flex);
	iter_subopt.push_back((double) node->sum_of_costs / (double) cleanup_head_lb);
	iter_sum_ll_generate.push_back(node->ll_generated);

	assert(num_of_agents > 0);
	if (iter_ag_lb.empty())
		iter_ag_lb = vector<vector<int>>(num_of_agents);
	if (iter_ag_cost.empty())
		iter_ag_cost = vector<vector<int>>(num_of_agents);
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		iter_ag_lb[ag].push_back(min_f_vals[ag]);
		iter_ag_cost[ag].push_back(paths[ag]->size()-1);
	}

	iter_use_flex.push_back(node->use_flex);
	iter_no_more_flex.push_back(node->no_more_flex);
	iter_cannot_use_flex.push_back(node->cannot_use_flex);
	if (node->chosen_from == "cleanup")
		iter_node_type.push_back(0);
	else if (node->chosen_from == "open")
		iter_node_type.push_back(1);
	else if (node->chosen_from == "focal")
		iter_node_type.push_back(2);
}


void IterTracker::getBranchStats(HLNode *node, int cleanup_head_lb,
	const vector<int>& init_min_f_vals, const vector<int>& init_costs)
{
	assert(br_sum_lb.empty());
	assert(br_sum_fval.empty());
	assert(br_sum_cost.empty());
	assert(br_num_conflicts.empty());
	assert(br_remained_flex.empty());

	if (br_ag_lb.empty())
		br_ag_lb = vector<vector<int>>(num_of_agents);
	if (br_ag_cost.empty())
		br_ag_cost = vector<vector<int>>(num_of_agents);

	int node_cnt = 0;
	while (node != nullptr)
	{
		int total_num_conf = (int)node->conflicts.size() + (int)node->unknownConf.size();
		double remain_flex = suboptimality * (double)node->g_val - (double)node->sum_of_costs;

		br_node_idx.push_back(node->time_generated);
		br_sum_lb.push_back(node->g_val);
		br_sum_fval.push_back(node->getFVal());
		br_sum_cost.push_back(node->sum_of_costs);
		br_num_conflicts.push_back(total_num_conf);
		br_remained_flex.push_back(remain_flex);
		if (cleanup_head_lb != 0)
			br_subopt.push_back((double)node->sum_of_costs / (double)cleanup_head_lb);

		vector<bool> set_val = vector<bool>(num_of_agents, false);
		list<pair<int, int>> tmp_ag_lb = node->getLBs();
		for (const auto& tmp_lb : tmp_ag_lb)
		{
			br_ag_lb[tmp_lb.first].push_back(tmp_lb.second);
			set_val[tmp_lb.first] = true;
		}
		for (int ag = 0; ag < num_of_agents; ag++)
			if (!set_val[ag])
				br_ag_lb[ag].push_back(-1);

		set_val = vector<bool>(num_of_agents, false);
		list<pair<int, int>> tmp_ag_cost = node->getCosts();
		for (const auto& tmp_cost : tmp_ag_cost)
		{
			br_ag_cost[tmp_cost.first].push_back(tmp_cost.second);
			set_val[tmp_cost.first] = true;
		}
		for (int ag = 0; ag < num_of_agents; ag++)
			if (!set_val[ag])
				br_ag_cost[ag].push_back(-1);

		br_sum_ll_generate.push_back(node->ll_generated);
		node = node->parent;
		node_cnt ++;
	}

	// Push back initial lowerbounds and costs of agents
	node_cnt ++;
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		br_ag_lb[ag].push_back(init_min_f_vals[ag]);
		br_ag_cost[ag].push_back(init_costs[ag]);
	}

	// Reverse the vectors
	std::reverse(br_node_idx.begin(), br_node_idx.end());
	std::reverse(br_sum_lb.begin(), br_sum_lb.end());
	std::reverse(br_sum_cost.begin(), br_sum_cost.end());
	std::reverse(br_num_conflicts.begin(), br_num_conflicts.end());
	std::reverse(br_remained_flex.begin(), br_remained_flex.end());
	std::reverse(br_subopt.begin(), br_subopt.end());
	std::reverse(br_sum_ll_generate.begin(), br_sum_ll_generate.end());
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		std::reverse(br_ag_lb[ag].begin(), br_ag_lb[ag].end());
		std::reverse(br_ag_cost[ag].begin(), br_ag_cost[ag].end());
	}

	// Fill the lowerbound and costs of agents
	for (int ag = 0; ag < num_of_agents; ag++)
		for (int i = 0; i < br_ag_lb[ag].size()-1; i++)
			if (br_ag_lb[ag].at(i+1) == -1)
				br_ag_lb[ag].at(i+1) = br_ag_lb[ag].at(i);
	for (int ag = 0; ag < num_of_agents; ag++)
		for (int i = 0; i < br_ag_cost[ag].size()-1; i++)
			if (br_ag_cost[ag].at(i+1) == -1)
				br_ag_cost[ag].at(i+1) = br_ag_cost[ag].at(i);
	return;
}


void IterTracker::getAllStats(HLNode *node)
{
	all_node_idx.push_back(node->time_generated);
	all_sum_lb.push_back(node->g_val);
	all_sum_fval.push_back(node->getFVal());
	all_sum_cost.push_back(node->sum_of_costs);
	all_num_conflicts.push_back(node->conflicts.size() + node->unknownConf.size());
	all_remained_flex.push_back(suboptimality * node->getFVal() - node->sum_of_costs);
	all_subopt.push_back((double) node->sum_of_costs / (double) node->getFVal());
	all_sum_ll_generate.push_back(node->ll_generated);
}


void IterTracker::saveIterStats(const string& file_path)
{
    ofstream stats;
    stats.open(file_path, std::ios::out);
    if (!stats.is_open())
    {
        cerr << "Failed to open file in IterTracker::saveIterStats.\n";
        return;
    }

    stats << "iter_sum_lb,";
	std::copy(iter_sum_lb.begin(), iter_sum_lb.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "iter_sum_fval,";
	std::copy(iter_sum_fval.begin(), iter_sum_fval.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "iter_sum_cost,";
	std::copy(iter_sum_cost.begin(), iter_sum_cost.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "iter_num_conflicts,";
	std::copy(iter_num_conflicts.begin(), iter_num_conflicts.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "iter_remained_flex,";
	std::copy(iter_remained_flex.begin(), iter_remained_flex.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "iter_subopt,";
	std::copy(iter_subopt.begin(), iter_subopt.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "iter_sum_ll_generate,";
	std::copy(iter_sum_ll_generate.begin(), iter_sum_ll_generate.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;
	stats << "replan_ll_generate,";
	std::copy(replan_ll_generate.begin(), replan_ll_generate.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;
	stats << "replan_agent,";
	std::copy(replan_agent.begin(), replan_agent.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;
	stats << "replan_flex,";
	std::copy(replan_flex.begin(), replan_flex.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;
	
	stats << "br_sum_lb,";
	std::copy(br_sum_lb.begin(), br_sum_lb.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "br_sum_fval,";
	std::copy(br_sum_fval.begin(), br_sum_fval.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "br_sum_cost,";
	std::copy(br_sum_cost.begin(), br_sum_cost.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "br_num_conflicts,";
	std::copy(br_num_conflicts.begin(), br_num_conflicts.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "br_remained_flex,";
	std::copy(br_remained_flex.begin(), br_remained_flex.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "br_subopt,";
	std::copy(br_subopt.begin(), br_subopt.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "br_sum_ll_generate,";
	std::copy(br_sum_ll_generate.begin(), br_sum_ll_generate.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;

	stats << "all_sum_lb,";
	std::copy(all_sum_lb.begin(), all_sum_lb.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "all_sum_fval,";
	std::copy(all_sum_fval.begin(), all_sum_fval.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "all_sum_cost,";
	std::copy(all_sum_cost.begin(), all_sum_cost.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "all_num_conflicts,";
	std::copy(all_num_conflicts.begin(), all_num_conflicts.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "all_remained_flex,";
	std::copy(all_remained_flex.begin(), all_remained_flex.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "all_subopt,";
	std::copy(all_subopt.begin(), all_subopt.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;
	stats << "all_sum_ll_generate,";
	std::copy(all_sum_ll_generate.begin(), all_sum_ll_generate.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
	stats << endl;

	stats << "open_sum_lb,";
	std::copy(open_sum_lb.begin(), open_sum_lb.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "open_sum_fval,";
	std::copy(open_sum_fval.begin(), open_sum_fval.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "open_sum_cost,";
	std::copy(open_sum_cost.begin(), open_sum_cost.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "open_num_conflicts,";
	std::copy(open_num_conflicts.begin(), open_num_conflicts.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "open_remained_flex,";
	std::copy(open_remained_flex.begin(), open_remained_flex.end(),
		std::ostream_iterator<double>(stats, ","));
	stats << endl;

	stats << "iter_node_idx,";
	std::copy(iter_node_idx.begin(), iter_node_idx.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "br_node_idx,";
	std::copy(br_node_idx.begin(), br_node_idx.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;
	stats << "open_node_idx,";
	std::copy(open_node_idx.begin(), open_node_idx.end(),
		std::ostream_iterator<int>(stats, ","));
	stats << endl;

	stats << "iter_ag_lb," << endl;
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		stats << ag << ",";
		std::copy(iter_ag_lb[ag].begin(), iter_ag_lb[ag].end(),
			std::ostream_iterator<int>(stats, ","));
		stats << endl;
	}
	stats << "iter_ag_cost," << endl;
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		stats << ag << ",";
		std::copy(iter_ag_cost[ag].begin(), iter_ag_cost[ag].end(), 
			std::ostream_iterator<int>(stats, ","));
		stats << endl;
	}
	stats << "br_ag_lb," << endl;
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		stats << ag << ",";
		std::copy(br_ag_lb[ag].begin(), br_ag_lb[ag].end(),
			std::ostream_iterator<int>(stats, ","));
		stats << endl;
	}
	stats << "br_ag_cost," << endl;
	for (int ag = 0; ag < num_of_agents; ag++)
	{
		stats << ag << ",";
		std::copy(br_ag_cost[ag].begin(), br_ag_cost[ag].end(), 
			std::ostream_iterator<int>(stats, ","));
		stats << endl;
	}

	stats.close();
    return;
}

void IterTracker::saveNumNodesInLists(const string& file_path)
{
    ofstream stats;
    stats.open(file_path, std::ios::out);
    if (!stats.is_open())
    {
        cerr << "Failed to open file in IterTracker::saveNumNodesInLists.\n";
        return;
    }
    stats << "iter_num_focal,";
    std::copy(iter_num_focal.begin(), iter_num_focal.end(),
        std::ostream_iterator<uint64_t>(stats, ","));
    stats << endl;
    stats << "iter_num_open,";
    std::copy(iter_num_open.begin(), iter_num_open.end(),
        std::ostream_iterator<uint64_t>(stats, ","));
    stats << endl;
    stats << "iter_num_cleanup,";
    std::copy(iter_num_cleanup.begin(), iter_num_cleanup.end(),
		std::ostream_iterator<uint64_t>(stats, ","));
    stats << endl;
    stats << "iter_node_type,";
    std::copy(iter_node_type.begin(), iter_node_type.end(),
		std::ostream_iterator<int>(stats, ","));
    stats << endl;
    stats << "iter_use_flex,";
    std::copy(iter_use_flex.begin(), iter_use_flex.end(),
		std::ostream_iterator<int>(stats, ","));
    stats << endl;
    stats << "iter_no_more_flex,";
    std::copy(iter_no_more_flex.begin(), iter_no_more_flex.end(),
		std::ostream_iterator<int>(stats, ","));
    stats << endl;
    stats << "iter_cannot_use_flex,";
    std::copy(iter_cannot_use_flex.begin(), iter_cannot_use_flex.end(),
		std::ostream_iterator<int>(stats, ","));
    stats << endl;
    stats << "iter_node_idx,";
    std::copy(iter_node_idx.begin(), iter_node_idx.end(),
		std::ostream_iterator<int>(stats, ","));
    stats << endl;

    stats.close();
    return;
}