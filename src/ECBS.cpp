#include "ECBS.h"


bool ECBS::solve(double time_limit, int _cost_lowerbound, int _cost_upperbound)
{
	this->cost_lowerbound = _cost_lowerbound;
	this->inadmissible_cost_lowerbound = 0;
	this->time_limit = time_limit;

	if (screen > 0) // 1 or 2
	{
		string name = getSolverName();
		name.resize(35, ' ');
		cout << name << ": ";
	}

	if (!is_start)  // set timer for restart with EECBS
	{
		is_start = true;
		start = clock();
	}

	generateRoot();

	while (!cleanup_list.empty() && !solution_found)
	{
		// Debug
		auto cleanup_head = cleanup_list.top();
		cleanup_head_lb = cleanup_list.top()->g_val;
		if (screen > 3)
			iter_tracker.getCleanupHeadStats(cleanup_head,
				focal_list.size(), open_list.size(), cleanup_list.size());
		// End debug

		auto curr = selectNode();  // update focal list and select the CT node

		// Debug
		assert((double) curr->sum_of_costs <= suboptimality * curr->getFVal());
		assert((double) curr->sum_of_costs <= suboptimality * cleanup_head->getFVal());

		if (screen > 3)
			iter_tracker.getIterStats(curr, cleanup_head_lb, min_f_vals, paths);
		// End debug

		if (terminate(curr))
			return solution_found;

		if (use_flex && curr->chosen_from == "cleanup")  // Early replanning for FEECBS
		{
			restart_cnt ++;
			if (restart_cnt > restart_th)
			{
				restart = true;
				break;
			}
		}

		if ((curr == dummy_start || curr->chosen_from == "cleanup") &&
		     !curr->h_computed) // heuristics has not been computed yet
		{
            runtime = (double)(clock() - start) / CLOCKS_PER_SEC;
            bool succ = heuristic_helper.computeInformedHeuristics(*curr, min_f_vals, time_limit - runtime);
            runtime = (double)(clock() - start) / CLOCKS_PER_SEC;
            if (!succ) // no solution, so prune this node
            {
                if (screen > 1)
                    cout << "	Prune " << *curr << endl;
                curr->clear();
                continue;
            }

            if (reinsertNode(curr))
                continue;
		}

        classifyConflicts(*curr);

		//Expand the node
		num_HL_expanded++;
		curr->time_expanded = num_HL_expanded;
		if (bypass && curr->chosen_from != "cleanup")
		{
			bool foundBypass = true;
			while (foundBypass)
			{
				if (terminate(curr))
					return solution_found;
				foundBypass = false;
				ECBSNode* child[2] = { new ECBSNode() , new ECBSNode() };

				curr->conflict = chooseConflict(*curr);
				addConstraints(curr, child[0], child[1]);
				if (screen > 1)
					cout << "	Expand " << *curr << endl << 	"	on " << *(curr->conflict) << endl;

				bool solved[2] = { false, false };
				vector<Path*> path_copy(paths);
				vector<int> fmin_copy(min_f_vals);
				for (int i = 0; i < 2; i++)
				{
					if (i > 0)
					{
						paths = path_copy;
						min_f_vals = fmin_copy;
					}
					solved[i] = generateChild(child[i], curr, i);
					if (!solved[i])
					{
						delete (child[i]);
						continue;
					}
					else if (i == 1 && !solved[0])
						continue;
					else if (bypass &&
						child[i]->sum_of_costs <= suboptimality * cost_lowerbound &&
						child[i]->distance_to_go < curr->distance_to_go) // Bypass1
					{
						foundBypass = true;

						assert(curr->g_val <= child[i]->g_val);

						if (!use_flex)  // bypass condition checking for ECBS and EECBS
						{
							for (const auto& path : child[i]->paths)
							{
								//// path.first: agent id
								//// path.second.first: path
								//// path.second.second: lower bound of the path
								if ((double)path.second.first.size()-1 > suboptimality * fmin_copy[path.first])
								{
									foundBypass = false;  // bypassing for EECBS
									break;
								}
							}
						}
						else  // bypass condition checking for FECBS and FEECBS
						{
							// if (curr->g_val < child[i]->g_val)  // do not bypass if the sum of lower bound increases
							// 	foundBypass = false;
							if (child[i]->sum_of_costs > suboptimality * curr->g_val)
								foundBypass = false;
						}

						if (foundBypass)
						{
							adoptBypass(curr, child[i], fmin_copy, path_copy);
							if (screen > 1)
								cout << "	Update " << *curr << endl;
							break;  // do not generate another child
						}
					}
				}
				if (foundBypass)
				{
					for (auto & i : child)
						delete i;
                    classifyConflicts(*curr); // classify the new-detected conflicts
				}

				else  // expansion
				{
					for (int i = 0; i < 2; i++)
					{
						if (solved[i])
						{
							pushNode(child[i]);
							curr->children.push_back(child[i]);
							if (screen > 1)
								cout << "		Generate " << *child[i] << endl;
							if (screen > 3)
								iter_tracker.getAllStats(child[i]);
						}
					}
				}
			}
		}
		else // no bypass
		{
			ECBSNode* child[2] = { new ECBSNode() , new ECBSNode() };
			curr->conflict = chooseConflict(*curr);
			
			addConstraints(curr, child[0], child[1]);

			if (screen > 1)
				cout << "	Expand " << *curr << endl << "	on " << *(curr->conflict) << endl;

			bool solved[2] = { false, false };
			vector<vector<PathEntry>*> path_copy(paths);
			vector<int> fmin_copy(min_f_vals);
			for (int i = 0; i < 2; i++)
			{
				if (i > 0)
				{
					paths = path_copy;
					min_f_vals = fmin_copy;
				}
				solved[i] = generateChild(child[i], curr, i);
				if (!solved[i])
				{
					delete (child[i]);
					continue;
				}
				pushNode(child[i]);
				curr->children.push_back(child[i]);
				if (screen > 1)
					cout << "		Generate " << *child[i] << endl;
				if (screen > 3)
					iter_tracker.getAllStats(child[i]);
			}
		}
		switch (curr->conflict->type)
		{
		case conflict_type::RECTANGLE:
			num_rectangle_conflicts++;
			break;
		case conflict_type::CORRIDOR:
			num_corridor_conflicts++;
			break;
		case  conflict_type::TARGET:
			num_target_conflicts++;
			break;
		case conflict_type::STANDARD:
			num_standard_conflicts++;
			break;
		case conflict_type::MUTEX:
			num_mutex_conflicts++;
			break;
		default:
			break;
		}
		if (curr->chosen_from == "cleanup")
			num_cleanup++;
		else if (curr->chosen_from == "open")
			num_open++;
		else if (curr->chosen_from == "focal")
			num_focal++;

		if (curr->conflict->priority == conflict_priority::CARDINAL)
			num_cardinal_conflicts++;
        if (!curr->children.empty())
            heuristic_helper.updateOnlineHeuristicErrors(*curr); // update online heuristic errors

		curr->clear();
	}  // end of while loop

	// Restart from FEECBS to EECBS
	runtime = (double)(clock() - start) / CLOCKS_PER_SEC;
	if (restart && runtime < time_limit)
	{
		clear();
		assert(use_flex);
		use_flex = false;
		srand(0);
		bool debug_foundSol = solve(time_limit, 0);
		assert(debug_foundSol == solution_found);
	}
	else if (runtime > time_limit && screen > 0)
		printResults();

	return solution_found;
}

void ECBS::adoptBypass(ECBSNode* curr, ECBSNode* child, const vector<int>& fmin_copy, const vector<Path*>& path_copy)
{
	num_adopt_bypass++;
	curr->sum_of_costs = child->sum_of_costs;
	curr->cost_to_go = child->cost_to_go;
	curr->distance_to_go = child->distance_to_go;
	curr->conflicts = child->conflicts;
	curr->unknownConf = child->unknownConf;
	curr->conflict = nullptr;
	curr->makespan = child->makespan;
	for (const auto& path : child->paths) // update paths
	{
		auto p = curr->paths.begin();
		while (p != curr->paths.end())
		{
			if (path.first == p->first)  // if path is already stored in curr node
			{
				assert(fmin_copy[path.first] == p->second.second);
				assert(path_copy[path.first]->size() - 1 == p->second.first.size() - 1);
				assert(path_copy[path.first] == &p->second.first);

				p->second.first = path.second.first;  // replace the path in curr node with path in child CT node
				paths[p->first] = &p->second.first;  // update the new path to global paths
                min_f_vals[p->first] = p->second.second;  // update back the old fmin to global
				break;
			}
			++p;
		}
		if (p == curr->paths.end())  // if path is not stored in curr node (e.g. root)
		{
			curr->paths.emplace_back(path);
			curr->paths.back().second.second = fmin_copy[path.first];
			paths[path.first] = &curr->paths.back().second.first;
			min_f_vals[path.first] = fmin_copy[path.first];  // update back to curr fmin
		}
	}
}

// takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
// also, do the same for ll_min_f_vals and paths_costs (since its already "on the way").
void ECBS::updatePaths(ECBSNode* curr)
{
	for (int i = 0; i < num_of_agents; i++)
	{
		paths[i] = &paths_found_initially[i].first;
		min_f_vals[i] = paths_found_initially[i].second;
	}
	vector<bool> updated(num_of_agents, false);  // initialized for false

	while (curr != nullptr)
	{
		for (auto & path : curr->paths)
		{
			int agent = path.first;
			if (!updated[agent])
			{
				paths[agent] = &path.second.first;
				min_f_vals[agent] = path.second.second;
				updated[agent] = true;
			}
		}
		curr = curr->parent;
	}
}


bool ECBS::generateRoot()
{
	paths.resize(num_of_agents, nullptr);
	init_min_f_vals.resize(num_of_agents);  // This is for debug
	mdd_helper.init(num_of_agents);
	heuristic_helper.init();

	// initialize paths_found_initially and min_f_vals
	paths_found_initially.clear();
	paths_found_initially.resize(num_of_agents);
	min_f_vals.clear();
	min_f_vals.resize(num_of_agents);

	auto root = new ECBSNode();
	root->g_val = 0;
	root->sum_of_costs = 0;
	root->ag_ll_node = vector<uint64_t>(num_of_agents, 0);

	auto agents = shuffleAgents();  //generate random permuattion of agent indices
	for (const int& ag : agents)
	{
		paths_found_initially[ag] = search_engines[ag]->findSuboptimalPath(*root, initial_constraints[ag], paths, ag, 0, suboptimality);
		if (paths_found_initially[ag].first.empty())
		{
			cout << "No path exists for agent " << ag << endl;
			return false;
		}
		paths[ag] = &paths_found_initially[ag].first;
		min_f_vals[ag] = paths_found_initially[ag].second;
		init_min_f_vals[ag] = paths_found_initially[ag].second;

		root->makespan = max(root->makespan, paths[ag]->size() - 1);
		root->g_val += min_f_vals[ag];
		root->sum_of_costs += (int)paths[ag]->size() - 1;
		root->ag_ll_node[ag] = search_engines[ag]->num_generated;

		num_LL_expanded += search_engines[ag]->num_expanded;
		num_LL_generated += search_engines[ag]->num_generated;
		num_LL_in_focal += search_engines[ag]->num_nodes_focal;
		num_findPathForSingleAgent ++;
		if (search_engines[ag]->use_focal)
			num_use_LL_focal ++;
		else
			num_use_LL_AStar ++;
	}

	root->h_val = 0;
	root->depth = 0;
	findConflicts(*root);
    heuristic_helper.computeQuickHeuristics(*root);
	pushNode(root);
	dummy_start = root;

	if (screen >= 2) // print start and goals
		printPaths();

	return true;
}


bool ECBS::generateChild(ECBSNode*  node, ECBSNode* parent, int child_idx)
{
	clock_t t1 = clock();
	node->parent = parent;
	node->HLNode::parent = parent;
	node->g_val = parent->g_val;
	node->h_val = parent->h_val;
	node->sum_of_costs = parent->sum_of_costs;
	node->makespan = parent->makespan;
	node->ag_ll_node = parent->ag_ll_node;
	node->depth = parent->depth + 1;
	auto agents = getInvalidAgents(node->constraints);
	assert(!agents.empty());
	for (auto agent : agents)
	{
		if (!findPathForSingleAgent(node, agent))
		{
			if (screen > 1)
				cout << "	No paths for agent " << agent << ". Node pruned." << endl;
			runtime_generate_child += (double)(clock() - t1) / CLOCKS_PER_SEC;
			return false;
		}
	}
	assert(node->sum_of_costs <= suboptimality * node->g_val);
	assert(parent->getFVal() <= node->getFVal());
	assert(parent->g_val <= node->g_val);

	findConflicts(*node);
	heuristic_helper.computeQuickHeuristics(*node);

	updateConflictImpacts(*node, *parent, child_idx);

	runtime_generate_child += (double)(clock() - t1) / CLOCKS_PER_SEC;
	return true;
}


bool ECBS::findPathForSingleAgent(ECBSNode*  node, int ag)
{
	clock_t t = clock();
	pair<Path, int> new_path;
	if (use_flex)
	{
		int other_sum_lb = node->g_val - min_f_vals[ag];
		int other_sum_cost = node->sum_of_costs - (int) paths[ag]->size() + 1;

		search_engines[ag]->setNodeLimit(node->ag_ll_node[ag]);  // Set limit to LL generated node

		if (use_flex_restriction)
		{
			// Flex restriction
			bool not_use_flex;
			if (node == dummy_start || node->parent == dummy_start)
			{
				not_use_flex = true;
			}
			else
			{
				not_use_flex = node->parent->chosen_from == "cleanup" || 
					node->parent->conflict->priority == conflict_priority::CARDINAL; //|| 
					// node->parent->g_val < node->g_val;
			}

			if (suboptimality* (double) other_sum_lb - (double) other_sum_cost >= 0 && not_use_flex)
			{
				// Not use flex if the CT node is from cleanup or the conflict is cardinal
				new_path = search_engines[ag]->findSuboptimalPath(*node, initial_constraints[ag], paths, ag, min_f_vals[ag], suboptimality);

				num_not_use_flex ++;
				node->use_flex = false;
				if (not_use_flex)
					node->cannot_use_flex = true;
			}

			else
			{
				new_path = search_engines[ag]->findSuboptimalPath(*node, initial_constraints[ag], paths, ag, min_f_vals[ag],
					suboptimality, other_sum_lb, other_sum_cost, init_sum_lb, flex, node->h_val);

				num_use_flex ++;
				node->use_flex = true;
				if (suboptimality* (double) other_sum_lb - (double) other_sum_cost < 0)
					node->no_more_flex = true;
			}
			// End Flex restrictions
		}
		else  // Without flex restriction
		{
			new_path = search_engines[ag]->findSuboptimalPath(*node, initial_constraints[ag], paths, ag,
				min_f_vals[ag], suboptimality, other_sum_lb, other_sum_cost, init_sum_lb, flex, node->h_val);	
		}
	}
	else
	{
		new_path = search_engines[ag]->findSuboptimalPath(*node, initial_constraints[ag], paths, ag, min_f_vals[ag], suboptimality);
	}

	if (screen > 3)  // This is for debugging
	{
		double replan_flx = suboptimality * node->g_val - min_f_vals[ag];
		replan_flx -= (node->sum_of_costs - (int) paths[ag]->size() + 1);
		iter_tracker.replan_flex.push_back(replan_flx);
		iter_tracker.replan_ll_generate.push_back(search_engines[ag]->num_generated);
		iter_tracker.replan_agent.push_back(ag);
	}

	node->ag_ll_node[ag] = search_engines[ag]->num_generated;  // update the node limit
	num_LL_expanded += search_engines[ag]->num_expanded;
	num_LL_generated += search_engines[ag]->num_generated;
	num_LL_in_focal += search_engines[ag]->num_nodes_focal;
	num_findPathForSingleAgent ++;
	if (search_engines[ag]->use_focal)
		num_use_LL_focal ++;
	else
		num_use_LL_AStar ++;

	runtime_build_CT += search_engines[ag]->runtime_build_CT;
	runtime_build_CAT += search_engines[ag]->runtime_build_CAT;
	runtime_path_finding += (double)(clock() - t) / CLOCKS_PER_SEC;
	if (new_path.first.empty())
		return false;
	assert(!isSamePath(*paths[ag], new_path.first));
	assert((int) new_path.first.size() - 1 >= max(new_path.second, min_f_vals[ag]));

	node->g_val = node->g_val + max(new_path.second - min_f_vals[ag], 0);
	if (node->parent != nullptr)
	    node->h_val = max(0, node->parent->getFVal() - node->g_val); // pathmax

	node->sum_of_costs = node->sum_of_costs - (int) paths[ag]->size() + (int) new_path.first.size();
	min_f_vals[ag] = max(new_path.second, min_f_vals[ag]);  // make sure the recorded lower bound is always the maximum
	new_path = make_pair(new_path.first, min_f_vals[ag]);
	node->paths.emplace_back(ag, new_path);
	paths[ag] = &node->paths.back().second.first;
	node->makespan = max(node->makespan, new_path.first.size() - 1);

	assert(node->sum_of_costs <= suboptimality * node->getFVal());
	assert(node->getFVal() >= node->parent->getFVal());

	return true;
}

inline void ECBS::pushNode(ECBSNode* node)
{
	num_HL_generated++;
	node->time_generated = num_HL_generated;
	// update handles
    node->cleanup_handle = cleanup_list.push(node);
	switch (solver_type)
	{
	case high_level_solver_type::ASTAREPS:  // cleanup_list is called open_list in ECBS
        if (node->sum_of_costs <= suboptimality * cost_lowerbound)
		{
			num_push_focal++;  // debug
            node->focal_handle = focal_list.push(node);
		}
		break;
	case high_level_solver_type::NEW:
		if (node->getFHatVal() <= suboptimality * cost_lowerbound)
		{
			node->focal_handle = focal_list.push(node);
			num_push_focal++;  // debug
		}
		break;
	case high_level_solver_type::EES:
		node->open_handle = open_list.push(node);
		if (node->getFHatVal() <= suboptimality * inadmissible_cost_lowerbound)
		{
			node->focal_handle = focal_list.push(node);
			num_push_focal++;  // debug
		}
		break;
	default:
		break;
	}
	allNodes_table.push_back(node);
}


inline bool ECBS::reinsertNode(ECBSNode* node)
{

	switch (solver_type)
	{
	case high_level_solver_type::ASTAREPS:
        if (node->sum_of_costs <= suboptimality * cost_lowerbound)
            return false;
        node->cleanup_handle = cleanup_list.push(node);
        break;
	case high_level_solver_type::NEW:
        if (node->getFHatVal() <= suboptimality * cost_lowerbound)
            return false;
        node->cleanup_handle = cleanup_list.push(node);
		break;
	case high_level_solver_type::EES:
        node->cleanup_handle = cleanup_list.push(node);
		node->open_handle = open_list.push(node);
		if (node->getFHatVal() <= suboptimality * inadmissible_cost_lowerbound)
			node->focal_handle = focal_list.push(node);
		break;
	default:
		break;
	}
	if (screen == 2)
	{
		cout << "	Reinsert " << *node << endl;
	}
	return true;
}


ECBSNode* ECBS::selectNode()
{
	ECBSNode* curr = nullptr;
	assert(solver_type != high_level_solver_type::ASTAR);
	switch (solver_type)
	{
	case high_level_solver_type::EES:
		// update the focal list if necessary
		if (open_list.top()->getFHatVal() != inadmissible_cost_lowerbound)
		{
			inadmissible_cost_lowerbound = open_list.top()->getFHatVal();
			double focal_list_threshold = suboptimality * inadmissible_cost_lowerbound;
			focal_list.clear();
			for (auto n : open_list)
			{
				if (n->getFHatVal() <= focal_list_threshold)
					n->focal_handle = focal_list.push(n);
			}
		}

		// choose the best node
		if (screen > 1 && cleanup_list.top()->getFVal() > cost_lowerbound)
			cout << "Lowerbound increases from " << cost_lowerbound << " to " << cleanup_list.top()->getFVal() << endl;
		cost_lowerbound = max(cleanup_list.top()->getFVal(), cost_lowerbound);
		if (focal_list.top()->sum_of_costs <= suboptimality * cost_lowerbound)
		{ // return best d
			curr = focal_list.top();
			curr->chosen_from = "focal";
			/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
			curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
			curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;
			curr->f_of_best_in_open = open_list.top()->getFVal();
			curr->f_hat_of_best_in_open = open_list.top()->getFHatVal();
			curr->d_of_best_in_open = open_list.top()->distance_to_go;
			curr->f_of_best_in_focal = focal_list.top()->getFVal();
			curr->f_hat_of_best_in_focal = focal_list.top()->getFHatVal();
			curr->d_of_best_in_focal = focal_list.top()->distance_to_go; */
			focal_list.pop();
			cleanup_list.erase(curr->cleanup_handle);
			open_list.erase(curr->open_handle);
		}
		else if (open_list.top()->sum_of_costs <= suboptimality * cost_lowerbound)
		{ // return best f_hat
			curr = open_list.top();
			curr->chosen_from = "open";
			/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
			curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
			curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;
			curr->f_of_best_in_open = open_list.top()->getFVal();
			curr->f_hat_of_best_in_open = open_list.top()->getFHatVal();
			curr->d_of_best_in_open = open_list.top()->distance_to_go;
			curr->f_of_best_in_focal = focal_list.top()->getFVal();
			curr->f_hat_of_best_in_focal = focal_list.top()->getFHatVal();
			curr->d_of_best_in_focal = focal_list.top()->distance_to_go;*/
			open_list.pop();
			cleanup_list.erase(curr->cleanup_handle);
			focal_list.erase(curr->focal_handle);
		}
		else
		{ // return best f
			curr = cleanup_list.top();
			curr->chosen_from = "cleanup";
			/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
			curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
			curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;
			curr->f_of_best_in_open = open_list.top()->getFVal();
			curr->f_hat_of_best_in_open = open_list.top()->getFHatVal();
			curr->d_of_best_in_open = open_list.top()->distance_to_go;
			curr->f_of_best_in_focal = focal_list.top()->getFVal();
			curr->f_hat_of_best_in_focal = focal_list.top()->getFHatVal();
			curr->d_of_best_in_focal = focal_list.top()->distance_to_go;*/
			cleanup_list.pop();
			open_list.erase(curr->open_handle);
			if (curr->getFHatVal() <= suboptimality * inadmissible_cost_lowerbound)
				focal_list.erase(curr->focal_handle);
		}
		break;
	case high_level_solver_type::ASTAREPS:
		// update the focal list if necessary
		if (cleanup_list.top()->getFVal() > cost_lowerbound)
		{
			if (screen == 3)
			{
				cout << "  Note -- FOCAL UPDATE!! from |FOCAL|=" << focal_list.size() << " with |OPEN|=" << cleanup_list.size() << " to |FOCAL|=";
			}
			double old_focal_list_threshold = suboptimality * cost_lowerbound;
			cost_lowerbound = max(cost_lowerbound, cleanup_list.top()->getFVal());
			double new_focal_list_threshold = suboptimality * cost_lowerbound;
			for (auto n : cleanup_list)
			{
				if (n->sum_of_costs > old_focal_list_threshold && n->sum_of_costs <= new_focal_list_threshold)
					n->focal_handle = focal_list.push(n);
			}
			if (screen == 3)
			{
				cout << focal_list.size() << endl;
			}
		}

		// choose best d in the focal list
		curr = focal_list.top();
		curr->chosen_from = "focal";
		/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
		curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
		curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;
		curr->f_of_best_in_focal = focal_list.top()->getFVal();
		curr->f_hat_of_best_in_focal = focal_list.top()->getFHatVal();
		curr->d_of_best_in_focal = focal_list.top()->distance_to_go;*/
		focal_list.pop();
		cleanup_list.erase(curr->cleanup_handle);
		break;
	case high_level_solver_type::NEW:
		// update the focal list if necessary
		if (cleanup_list.top()->getFVal() > cost_lowerbound)
		{
			if (screen == 3)
			{
				cout << "  Note -- FOCAL UPDATE!! from |FOCAL|=" << focal_list.size() << " with |OPEN|=" << cleanup_list.size() << " to |FOCAL|=";
			}
			double old_focal_list_threshold = suboptimality * cost_lowerbound;
			cost_lowerbound = max(cost_lowerbound, cleanup_list.top()->getFVal());
			double new_focal_list_threshold = suboptimality * cost_lowerbound;
			focal_list.clear();
			for (auto n : cleanup_list)
			{
				heuristic_helper.updateInadmissibleHeuristics(*n);
				if (n->getFHatVal() <= new_focal_list_threshold)
					n->focal_handle = focal_list.push(n);
			}
			if (screen == 3)
			{
				cout << focal_list.size() << endl;
			}
		}

		if (focal_list.empty()) // choose best f in the cleanup list (to improve the lower bound)
		{
			curr = cleanup_list.top();
			curr->chosen_from = "cleanup";
			/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
			curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
			curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;*/
			cleanup_list.pop();
		}
		else // choose best d in the focal list
		{
			curr = focal_list.top();
			curr->chosen_from = "focal";
			/*curr->f_of_best_in_cleanup = cleanup_list.top()->getFVal();
			curr->f_hat_of_best_in_cleanup = cleanup_list.top()->getFHatVal();
			curr->d_of_best_in_cleanup = cleanup_list.top()->distance_to_go;
			curr->f_of_best_in_focal = focal_list.top()->getFVal();
			curr->f_hat_of_best_in_focal = focal_list.top()->getFHatVal();
			curr->d_of_best_in_focal = focal_list.top()->distance_to_go;*/
			focal_list.pop();
			cleanup_list.erase(curr->cleanup_handle);
		}
		break;
	default:
		break;
	}

	// takes the paths_found_initially and UPDATE all constrained paths found for agents from curr to dummy_start (and lower-bounds)
	updatePaths(curr);

	if (screen > 1)
		cout << endl << "Pop " << *curr << endl;
	return curr;
}

void ECBS::printPaths() const
{
	for (int i = 0; i < num_of_agents; i++)
	{
		cout << "Agent " << i << " (" << paths_found_initially[i].first.size() - 1 << " -->" <<
			paths[i]->size() - 1 << "): ";
		for (const auto & t : *paths[i])
			cout << t.location << "->";
		cout << endl;
	}
}


void ECBS::classifyConflicts(ECBSNode &node)
{
    if (node.unknownConf.empty())
        return;
	// Classify all conflicts in unknownConf
	while (!node.unknownConf.empty())
	{
		shared_ptr<Conflict> con = node.unknownConf.front();
		int a1 = con->a1, a2 = con->a2;
		int timestep = get<3>(con->constraint1.back());
		constraint_type type = get<4>(con->constraint1.back());
		node.unknownConf.pop_front();

		if (PC)
		    if (node.chosen_from == "cleanup" ||
               // (min_f_vals[a1] * suboptimality >= min_f_vals[a1] + 1 &&
               //min_f_vals[a2] * suboptimality >= min_f_vals[a2] + 1))
               (int)paths[a1]->size() - 1 == min_f_vals[a1] ||
               (int)paths[a2]->size() - 1 == min_f_vals[a2]) // the path of at least one agent is its shortest path
			    computeConflictPriority(con, node);

		/*if (con->priority == conflict_priority::CARDINAL && heuristic_helper.type == heuristics_type::ZERO)
		{
			computeSecondPriorityForConflict(*con, node);
			node.conflicts.push_back(con);
			return;
		}*/

		// TODO: Mutex reasoning for ECBS
		/*if (mutex_reasoning)
		{
			// TODO mutex reasoning is per agent pair, don't do duplicated work...
			auto mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
			auto mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());

			auto mutex_conflict = mutex_helper.run(a1, a2, node, mdd1, mdd2);

			if (mutex_conflict != nullptr)
			{
				computeSecondPriorityForConflict(*mutex_conflict, node);
				node.conflicts.push_back(mutex_conflict);
				continue;
			}
		}*/

		// Target Reasoning: highest secondary priority
		if (con->type == conflict_type::TARGET)
		{
			computeSecondPriorityForConflict(*con, node);
			node.conflicts.push_back(con);
			continue;
		}

		// Corridor reasoning: second secondary priority
		if (corridor_reasoning)
		{
			auto corridor = corridor_helper.run(con, paths, node);
			if (corridor != nullptr)
			{
				corridor->priority = con->priority;
				computeSecondPriorityForConflict(*corridor, node);
				node.conflicts.push_back(corridor);
				continue;
			}
		}


		// Rectangle reasoning: less secondary priority
		if (rectangle_reasoning &&
		    (int)paths[a1]->size() - 1 == min_f_vals[a1] && // the paths for both agents are their shortest paths
		    (int)paths[a2]->size() - 1 == min_f_vals[a2] &&
            min_f_vals[a1] > timestep &&  //conflict happens before both agents reach their goal locations
			min_f_vals[a2] > timestep &&
			type == constraint_type::VERTEX) // vertex conflict
		{
			auto mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
			auto mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());
			auto rectangle = rectangle_helper.run(paths, timestep, a1, a2, mdd1, mdd2);
			if (rectangle != nullptr)
			{
                if (!PC)
                    rectangle->priority = conflict_priority::UNKNOWN;
				computeSecondPriorityForConflict(*rectangle, node);
				node.conflicts.push_back(rectangle);
				continue;
			}
		}

		computeSecondPriorityForConflict(*con, node);
		node.conflicts.push_back(con);
	}

	// remove conflicts that cannot be chosen, to save some memory
	removeLowPriorityConflicts(node.conflicts);
}


void ECBS::computeConflictPriority(shared_ptr<Conflict>& con, ECBSNode& node)
{
    int a1 = con->a1, a2 = con->a2;
	int timestep = get<3>(con->constraint1.back());
	constraint_type type = get<4>(con->constraint1.back());
	bool cardinal1 = false, cardinal2 = false;
	MDD *mdd1 = nullptr, *mdd2 = nullptr;
	if (timestep >= (int)paths[a1]->size())
		cardinal1 = true;
	else
	{
		mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
	}
	if (timestep >= (int)paths[a2]->size())
		cardinal2 = true;
	else
	{
		mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());
	}

	if (type == constraint_type::EDGE) // Edge conflict
	{
		if (timestep < (int)mdd1->levels.size())
		{
			cardinal1 = mdd1->levels[timestep].size() == 1 &&
				mdd1->levels[timestep].front()->location == paths[a1]->at(timestep).location &&
				mdd1->levels[timestep - 1].size() == 1 &&
				mdd1->levels[timestep - 1].front()->location == paths[a1]->at(timestep - 1).location;
		}
		if (timestep < (int)mdd2->levels.size())
		{
			cardinal2 = mdd2->levels[timestep].size() == 1 &&
				mdd2->levels[timestep].front()->location == paths[a2]->at(timestep).location &&
				mdd2->levels[timestep - 1].size() == 1 &&
				mdd2->levels[timestep - 1].front()->location == paths[a2]->at(timestep - 1).location;
		}
	}
	else // vertex conflict or target conflict
	{
		if (!cardinal1 && timestep < (int)mdd1->levels.size())
		{
			cardinal1 = mdd1->levels[timestep].size() == 1 &&
				mdd1->levels[timestep].front()->location == paths[a1]->at(timestep).location;
		}
		if (!cardinal2 && timestep < (int)mdd2->levels.size())
		{
			cardinal2 = mdd2->levels[timestep].size() == 1 &&
				mdd2->levels[timestep].front()->location == paths[a2]->at(timestep).location;
		}
	}

	if (cardinal1 && cardinal2)
	{
		con->priority = conflict_priority::CARDINAL;
	}
	else if (cardinal1 || cardinal2)
	{
		con->priority = conflict_priority::SEMI;
	}
	else
	{
		con->priority = conflict_priority::NON;
	}
}


// used for rapid random  restart
void ECBS::clear()
{
    mdd_helper.clear();
    heuristic_helper.clear();
    paths.clear();
    paths_found_initially.clear();
    min_f_vals.clear();
	init_min_f_vals.clear();

    open_list.clear();
    cleanup_list.clear();
    focal_list.clear();
    for (auto& node : allNodes_table)
        delete node;
    allNodes_table.clear();

	path_initialize = false;
    dummy_start = nullptr;
    goal_node = nullptr;
    solution_found = false;
    solution_cost = -2;
}