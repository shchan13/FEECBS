# FEECBS: EECBS with Flex Distribution

## Introduction
Explicit Estimation Conflict-Based Search (EECBS) is an effifent bounded-suboptimal algorithm for solving Multi-Agent Path Finding (MAPF) [1]. It uses Focal Search on the low level to find paths for agents individually and use Explicit Estimation Search (EES) to resolve collisions on the high level. In this work, we use the difference between the focal threshold and the path cost, defined as flex, to speed up the search on the low level. For more details, please check our paper in AAAI 2022 [2].

## How to Run the Code
The code requires the external library BOOST (https://www.boost.org/). After you installed BOOST and downloaded the source code, go into the directory of the source code and compile it with CMake: 
```
cmake .
make
```

You also need to download the MAPF instances from the MAPF benchmark (https://movingai.com/benchmarks/mapf/index.html).

Then, you are able to run the code:
```
./eecbs -m random-32-32-20.map -a random-32-32-20-random-1.scen -o test.csv --outputPaths=paths.txt -k 50 -t 60 --suboptimality=1.2 --flex false
```

- m: the map file from the MAPF benchmark
- a: the scenario file from the MAPF benchmark
- o: the output file that contains the search statistics
- outputPaths: the output file that contains the paths 
- k: the number of agents
- t: the runtime limit
- suboptimality: the suboptimality factor w
- flex: whether to use flex (*bool*, default: false)

You can find more details and explanations for all parameters with:
```
./eecbs --help
```

## License
EECBS is released under USC – Research License. See license.md for further details.
 
## References
[1] Jiaoyang Li, Wheeler Ruml and Sven Koenig.
EECBS: Bounded-Suboptimal Search for Multi-Agent Path Finding.
In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), (in print), 2021.

[2] Shao-Hung Chan, Jiaoyang Li, Graeme Gange, Daniel Harabor, Peter J. Stuckey, Sven Koenig. Flex Distribution for Bounded-Suboptimal Multi-Agent Path Finding. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), (in print), 2022.
