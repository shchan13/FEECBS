#pragma once
#include <tuple>
#include <list>
#include <vector>
#include <set>
#include <ctime>
#include <fstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

using boost::heap::pairing_heap;
using boost::heap::compare;
using boost::unordered_map;
using boost::unordered_set;
using std::vector;
using std::list;
using std::set;
using std::get;
using std::tuple;
using std::make_tuple;
using std::pair;
using std::make_pair;
using std::tie;
using std::min;
using std::max;
using std::shared_ptr;
using std::make_shared;
using std::clock;
using std::cout;
using std::endl;
using std::ofstream;
using std::cerr;
using std::string;
using std::stable_sort;

// #define NDEBUG 

#define MAX_TIMESTEP INT_MAX / 2
#define MAX_COST INT_MAX / 2
#define MAX_NODES INT_MAX / 2

struct PathEntry
{
	int location = -1;
	// bool single = false;
  // int mdd_width;

  // bool is_single() const {
  //  return mdd_width == 1;
  //}
	PathEntry(int loc = -1) { location = loc; }
};

typedef vector<PathEntry> Path;
std::ostream& operator<<(std::ostream& os, const Path& path);

bool isSamePath(const Path& p1, const Path& p2);

// Only for three-tuples of std::hash-able types for simplicity.
// You can of course template this struct to allow other hash functions
/*struct three_tuple_hash {
    template <class T1, class T2, class T3>
    std::size_t operator () (const std::tuple<T1, T2, T3> &p) const {
        auto h1 = std::hash<T1>{}(get<0>(p));
        auto h2 = std::hash<T2>{}(get<1>(p));
        auto h3 = std::hash<T3>{}(get<2>(p));
        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2 ^ h3;
    }
};*/

template <typename T, typename S>
vector<int> sort_indexes(const vector<T>& v, bool v_ascend, const vector<S>& u, bool u_ascend) 
{
    if (!u.empty() && v.size() != u.size())
    {
        cerr << "The size of vector v should be the same a the size of the vector u" << endl;
        exit(1);
    }

    // initialize original index locations
    vector<int> idx(v.size());
    for (int i = 0; i < (int) v.size(); i++)
        idx[i] = i;

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    if (!v.empty())
    {
        if (v_ascend)
            stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        else
            stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    }

    if (!u.empty())
    {
        if (u_ascend)
            stable_sort(idx.begin(), idx.end(), [&u](size_t i1, size_t i2) {return u[i1] < u[i2];});
        else
            stable_sort(idx.begin(), idx.end(), [&u](size_t i1, size_t i2) {return u[i1] > u[i2];});
    }

    return idx;
}

template <typename T>
vector<int> sort_indexes(const vector<T>& v, bool _ascending) 
{
    // initialize original index locations
    vector<int> idx(v.size());
    for (int i = 0; i < (int) v.size(); i++)
        idx[i] = i;

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    if (_ascending)
        stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    else
        stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

struct conflict_impact
{
    int increased_lb = 0;  // Averaged by the counts
    int reduced_num_conflicts = 0;  // Averaged by the counts
    double increased_flex = 0;  // w * node->getFVal() - node->sum_of_costs, which is averaged by the counts
    int count = 0;
};
