#pragma once
#include "CBSHeuristic.h"
#include "RectangleReasoning.h"
#include "CorridorReasoning.h"
#include "MutexReasoning.h"
#include "IterTracker.h"

enum high_level_solver_type { ASTAR, ASTAREPS, NEW, EES};

class CBS
{
public:
	bool randomRoot = false; // randomize the order of the agents in the root CT node
	bool path_initialize = false;  // Check if the paths are initialized by the outer solver

	/////////////////////////////////////////////////////////////////////////////////////
	// stats
	double runtime = 0;
	double runtime_generate_child = 0; // runtime of generating child nodes
	double runtime_build_CT = 0; // runtime of building constraint table
	double runtime_build_CAT = 0; // runtime of building conflict avoidance table
	double runtime_path_finding = 0; // runtime of finding paths for single agents
	double runtime_detect_conflicts = 0;
	double runtime_preprocessing = 0; // runtime of building heuristic table for the low level
	double runtime_solver = 0;  // runtime for using inner solver for meta-agents
	double runtime_sort_ma = 0;  // runtime for computing priority heuristic
	double runtime_merge = 0;  // runtime for merging the agents

	uint64_t num_cardinal_conflicts = 0;
	uint64_t num_corridor_conflicts = 0;
	uint64_t num_rectangle_conflicts = 0;
	uint64_t num_target_conflicts = 0;
	uint64_t num_mutex_conflicts = 0;
	uint64_t num_standard_conflicts = 0;

	uint64_t num_adopt_bypass = 0; // number of times when adopting bypasses
	uint64_t num_push_focal = 0;  // number of child CT node being pushed to FOCAL

	uint64_t num_HL_expanded = 0;
	uint64_t num_HL_generated = 0;
	uint64_t num_LL_expanded = 0;
	uint64_t num_LL_generated = 0;
	uint64_t num_LL_in_focal = 0;

	uint64_t solver_counter;
	uint64_t solver_num_HL_expanded = 0;
	uint64_t solver_num_HL_generated = 0;
	uint64_t solver_num_LL_expanded = 0;
	uint64_t solver_num_LL_generated = 0;

	uint64_t num_cleanup = 0; // number of expanded nodes chosen from cleanup list
	uint64_t num_open = 0; // number of expanded nodes chosen from open list
	uint64_t num_focal = 0; // number of expanded nodes chsoen from focal list
	uint64_t num_use_LL_AStar = 0;
	uint64_t num_use_LL_focal = 0;
	uint64_t num_not_use_flex = 0;
	uint64_t num_use_flex = 0;
	uint64_t num_findPathForSingleAgent = 0;

	HLNode* dummy_start = nullptr;
	HLNode* goal_node = nullptr;



	bool solution_found = false;
	int solution_cost = -2;

	/////////////////////////////////////////////////////////////////////////////////////////
	// set params
	void setHeuristicType(heuristics_type h, heuristics_type h_hat)
	{
	    heuristic_helper.type = h;
	    heuristic_helper.setInadmissibleHeuristics(h_hat);
	}
	void setPrioritizeConflicts(bool p) {PC = p;	heuristic_helper.PC = p; }
	void setRectangleReasoning(bool r) {rectangle_reasoning = r; heuristic_helper.rectangle_reasoning = r; }
	void setCorridorReasoning(bool c) {corridor_reasoning = c; heuristic_helper.corridor_reasoning = c; }
	void setTargetReasoning(bool t) {target_reasoning = t; heuristic_helper.target_reasoning = t; }
	void setMutexReasoning(bool m) {mutex_reasoning = m; heuristic_helper.mutex_reasoning = m; }
	void setDisjointSplitting(bool d) {disjoint_splitting = d; heuristic_helper.disjoint_splitting = d; }
	void setBypass(bool b) { bypass = b; } // 2-agent solver for heuristic calculation does not need bypass strategy.
	void setConflictSelectionRule(conflict_selection c) { conflict_selection_rule = c; heuristic_helper.conflict_selection_rule = c; }
	void setNodeSelectionRule(node_selection n) { node_selection_rule = n; heuristic_helper.node_selection_rule = n; }
	void setSavingStats(bool s) { save_stats = s; heuristic_helper.save_stats = s; }
	void setHighLevelSolver(high_level_solver_type s, double w)
	{
		solver_type = s;
		suboptimality = w;
	}
	void setNodeLimit(int n) { node_limit = n; }
	void setInitConstraints(int agent, ConstraintTable _table) {initial_constraints[agent].init(_table);}
	void setInitSumLB (int _sum_lb) { init_sum_lb = _sum_lb; }
	void setFlex(double val) {flex = val;}
	void setTimeLimit(double tl) { time_limit = tl; }
	void setIsStart(bool _st) { is_start = _st; }
	void setIsSolver(bool s) {is_solver = s;}
	void setMinFVal(int agent, int val)
	{
		if (min_f_vals.empty())
			min_f_vals.resize(num_of_agents);
		min_f_vals[agent] = val; 
	}
	virtual void setInitialPath(int agent, Path _path)
	{ 
		if (paths_found_initially.empty())
			paths_found_initially.resize(num_of_agents);
		paths_found_initially[agent] = _path; 
	}
	virtual int getInitialPathLength(int agent) const {return (int) paths_found_initially[agent].size() - 1; }
	int getminFVal(int agent) const { return min_f_vals[agent]; }
	void setRestartTh(int th) {restart_th = th;}
	void setUseFlex(bool _f) { use_flex = _f; }
	void setUseFlexRestriction(bool _f) {use_flex_restriction = _f;}

	////////////////////////////////////////////////////////////////////////////////////////////
	// Runs the algorithm until the problem is solved or time is exhausted 
	virtual bool solve(double time_limit, int _cost_lowerbound = 0, int _cost_upperbound = MAX_COST);

	int getLowerBound() const { return cost_lowerbound; }
	Path getPath(int agent) const { return *paths[agent]; }
	vector<Path> getPath(void) const;

	CBS(const Instance& instance, bool sipp, int screen);
	CBS(vector<SingleAgentSolver*>& search_engines,
		const vector<ConstraintTable>& constraints,
		vector<Path>& paths_found_initially, int screen);
	void clearSearchEngines();
	~CBS();

	// Save results
	void saveResults(const string &fileName, const string &instanceName) const;
	void saveStats(const string &fileName, const string &instanceName);
	void saveCT(const string &fileName) const; // write the CT to a file
    void savePaths(const string &fileName) const; // write the paths to a file
	virtual void clear(); // used for rapid random  restart

protected:
    bool rectangle_reasoning;  // using rectangle reasoning
	bool corridor_reasoning;  // using corridor reasoning
	bool target_reasoning;  // using target reasoning
	bool disjoint_splitting;  // disjoint splitting
	bool mutex_reasoning;  // using mutex reasoning
	bool bypass; // using Bypass1
	bool PC; // prioritize conflicts
	bool save_stats;
	high_level_solver_type solver_type; // the solver for the high-level search
	conflict_selection conflict_selection_rule;
	node_selection node_selection_rule;

	MDDTable mdd_helper;	
	RectangleReasoning rectangle_helper;
	CorridorReasoning corridor_helper;
	MutexReasoning mutex_helper;
	CBSHeuristic heuristic_helper;

	list<HLNode*> allNodes_table; // this is ued for both ECBS and EES


	string getSolverName() const;

	int screen;
	
	double time_limit;
	double suboptimality = 1.0;
	int cost_lowerbound = 0;
	int inadmissible_cost_lowerbound;
	int node_limit = MAX_NODES;
	int cost_upperbound = MAX_COST;
	int cleanup_head_lb;  // lower bound of the top node in CLEANUP

	vector<ConstraintTable> initial_constraints;
	clock_t start;

	int num_of_agents;

	vector<Path*> paths;
	// vector<MDD*> mdds_initially;  // contain initial paths found
	vector < SingleAgentSolver* > search_engines;  // used to find (single) agents' paths and mdd

	// For nested framework
	bool is_solver = false;
	bool is_start = false;
	int merge_th = INT_MAX;
	int init_sum_lb = 0;  // Obtain from outer (E)CBS, the initial cost_lower bound
	double flex = 0.0;  // flex for the meta-agent
	vector<int> min_f_vals; // lower bounds of the cost of the shortest path
	vector<int> init_min_f_vals; // lower bounds of the cost of the shortest path at the root CT node

	bool use_flex = false;  // Whether to use FEECBS or EECBS
	bool use_flex_restriction = false;  // Whether to use flex restriction
	bool restart = false;

	// For CLEANUP node slection
	int restart_th;
	int restart_cnt = 0;

	// For impact-based search
	bool use_impact;
	list<shared_ptr<Conflict>> seen_conflicts;  // conflicts being seen on other branches
	void updateConflictImpacts(const HLNode& node, const HLNode& parent, int child_idx);

	// Debug
	uint16_t num_use_priority = 0;
	uint16_t num_use_type = 0;
	uint16_t num_use_second_priority = 0;
	uint16_t num_use_increased_flex = 0;
	uint16_t num_use_increased_lb = 0;
	uint16_t num_use_reduced_conflicts = 0;
	uint16_t num_tie = 0;
	uint16_t num_has_seen_conf = 0;
	uint16_t num_use_count = 0;
	bool compareConflicts(shared_ptr<Conflict> conflict1, shared_ptr<Conflict> conflict2);
	shared_ptr<Conflict> debugChooseConflict(const HLNode &node);
	// End impact-based search

	void addConstraints(const HLNode* curr, HLNode* child1, HLNode* child2) const;
	set<int> getInvalidAgents(const list<Constraint>& constraints); // return agents that violates the constraints
	//conflicts
	void findConflicts(HLNode& curr);
	void findConflicts(HLNode& curr, int a1, int a2);
	shared_ptr<Conflict> chooseConflict(const HLNode &node) const;
	static void copyConflicts(const list<shared_ptr<Conflict>>& conflicts,
		list<shared_ptr<Conflict>>& copy, const list<int>& excluded_agent) ;
	void removeLowPriorityConflicts(list<shared_ptr<Conflict>>& conflicts) const;
	void computeSecondPriorityForConflict(Conflict& conflict, const HLNode& node);

	inline void releaseNodes();

	// print and save
	void printResults() const;
	static void printConflicts(const HLNode &curr) ;

	bool validateSolution() const;
	inline int getAgentLocation(int agent_id, size_t timestep) const;

	vector<int> shuffleAgents() const;  //generate random permuattion of agent indices
	bool terminate(HLNode* curr); // check the stop condition and return true if it meets
	void computeConflictPriority(shared_ptr<Conflict>& con, CBSNode& node); // check the conflict is cardinal, semi-cardinal or non-cardinal

	IterTracker iter_tracker;

	// void getBranchEval(HLNode* __node__, int open_head_lb);
	// void saveEval(void);
	// void saveNumNodesInLists(void);

private: // CBS only, cannot be used by ECBS
	vector<Path> paths_found_initially;  // contain initial paths found
	pairing_heap< CBSNode*, compare<CBSNode::compare_node_by_f> > cleanup_list; // it is called open list in ECBS
	pairing_heap< CBSNode*, compare<CBSNode::compare_node_by_inadmissible_f> > open_list; // this is used for EES
	pairing_heap< CBSNode*, compare<CBSNode::compare_node_by_d> > focal_list; // this is ued for both ECBS and EES

	// node operators
	inline void pushNode(CBSNode* node);
	CBSNode* selectNode();
	inline bool reinsertNode(CBSNode* node);

		 // high level search
	bool generateChild(CBSNode* child, CBSNode* curr, int child_idx=0);
	bool generateRoot();
	bool findPathForSingleAgent(CBSNode*  node, int ag, int lower_bound = 0);
	void classifyConflicts(CBSNode &parent);
		 //update information
	inline void updatePaths(CBSNode* curr);
	void printPaths() const;
};
