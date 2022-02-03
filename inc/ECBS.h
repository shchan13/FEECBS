#pragma once
#include "CBS.h"
#include "ECBSNode.h"


class ECBS : public CBS
{
public:
	ECBS(const Instance& instance, bool sipp, int screen) : CBS(instance, sipp, screen) {	
		if (screen > 3)
		{
			// Initialize for agents analysis
			iter_sum_lb = make_shared<vector<int>>();
			br_sum_lb = make_shared<vector<int>>();
    		all_sum_lb = make_shared<vector<int>>();
			open_sum_lb = make_shared<vector<int>>();

			iter_sum_fval = make_shared<vector<int>>();
			br_sum_fval = make_shared<vector<int>>();
    		all_sum_fval = make_shared<vector<int>>();
			open_sum_fval = make_shared<vector<int>>();

    		iter_sum_cost = make_shared<vector<int>>();
    		br_sum_cost = make_shared<vector<int>>();
    		all_sum_cost = make_shared<vector<int>>();
    		open_sum_cost = make_shared<vector<int>>();
    
			iter_num_conflicts = make_shared<vector<int>>();
			br_num_conflicts = make_shared<vector<int>>();
			all_num_conflicts = make_shared<vector<int>>();
			open_num_conflicts = make_shared<vector<int>>();

			iter_remained_flex = make_shared<vector<double>>();
			br_remained_flex = make_shared<vector<double>>();
			all_remained_flex = make_shared<vector<double>>();
			open_remained_flex = make_shared<vector<double>>();

			iter_subopt = make_shared<vector<double>>();
			br_subopt = make_shared<vector<double>>();
			all_subopt = make_shared<vector<double>>();

			iter_sum_ll_generate = make_shared<vector<uint64_t>>();
			br_sum_ll_generate = make_shared<vector<uint64_t>>();
			all_sum_ll_generate = make_shared<vector<uint64_t>>();
			replan_ll_generate = make_shared<vector<uint64_t>>();
			replan_agent = make_shared<vector<int>>();
			replan_flex = make_shared<vector<double>>();

			iter_node_idx = make_shared<vector<int>>();
			br_node_idx = make_shared<vector<int>>();
			all_node_idx = make_shared<vector<int>>();
			open_node_idx = make_shared<vector<int>>();

			iter_ag_lb = make_shared<vector<vector<int>>>(num_of_agents);
			br_ag_lb = make_shared<vector<vector<int>>>(num_of_agents);

			iter_ag_cost = make_shared<vector<vector<int>>>(num_of_agents);
			br_ag_cost = make_shared<vector<vector<int>>>(num_of_agents);
		}

		if (screen == 5)
		{
			iter_num_focal = make_shared<vector<uint64_t>>();
			iter_num_open = make_shared<vector<uint64_t>>();
			iter_num_cleanup = make_shared<vector<uint64_t>>();
			iter_node_type = make_shared<vector<int>>();
			iter_use_flex = make_shared<vector<bool>>();
			iter_no_more_flex = make_shared<vector<bool>>();
			iter_cannot_use_flex = make_shared<vector<bool>>();
		}
	}
	void setInitialPath(int agent, Path _path) override
	{ 
		if (paths_found_initially.empty())
			paths_found_initially.resize(num_of_agents);
		paths_found_initially[agent].first = _path;
		cout << paths_found_initially[agent].first << endl;
	}
	void setLLNodeLimitRatio(double lr)
	{
		for (int i = 0; i < num_of_agents; i++)
			search_engines[i]->setNodeLimitRatio(lr);
		return;
	}

	int getInitialPathLength(int agent) const override {return (int) paths_found_initially[agent].first.size() - 1; }

	////////////////////////////////////////////////////////////////////////////////////////////
	// Runs the algorithm until the problem is solved or time is exhausted 
	bool solve(double time_limit, int _cost_lowerbound = 0, int _cost_upperbound = MAX_COST) override;
    void clear() override; // used for rapid random  restart

private:
	vector< pair<Path, int> > paths_found_initially;  // contain initial paths found
	pairing_heap< ECBSNode*, compare<ECBSNode::compare_node_by_f> > cleanup_list; // it is called open list in ECBS
	pairing_heap< ECBSNode*, compare<ECBSNode::compare_node_by_inadmissible_f> > open_list; // this is used for EES
	pairing_heap< ECBSNode*, compare<ECBSNode::compare_node_by_d> > focal_list; // this is ued for both ECBS and EES

	void adoptBypass(ECBSNode* curr, ECBSNode* child, const vector<int>& fmin_copy, const vector<Path*>& path_copy);

	// node operators
	void pushNode(ECBSNode* node);
	ECBSNode* selectNode();
	bool reinsertNode(ECBSNode* node);

	// high level search
	bool generateChild(ECBSNode* child, ECBSNode* curr, int child_idx=0);
	bool generateRoot();
	bool findPathForSingleAgent(ECBSNode*  node, int ag);
	bool findPathForMetaAgent(ECBSNode* node, const vector<int>& meta_ag);
	void classifyConflicts(ECBSNode &node);
	void computeConflictPriority(shared_ptr<Conflict>& con, ECBSNode& node);

	// For NFECBS and NFEECBS
	void getFlex(const vector<int>& agent);

	//update information
	void updatePaths(ECBSNode* curr);
	void printPaths() const;
};