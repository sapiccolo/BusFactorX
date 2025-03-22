# BusFactorX

## Overview

**BusFactorX** is a Python library designed to facilitate the estimation of a project bus-factor.
The bus-factor is defined as the smallest nunmber of people that have to disappear from a project — as if they were hit by a bus — for the project to stall or experience a significant delay.
Turns out that computing the bus factor of a project is an NP-Hard problem. As such, BusFactorX offers efficient approximation algorithms to estimate the bus-factor of a project in reasonable time.

In BusFactorX, a project is modelled as a bipartite graph of people and tasks, represented as a scipy sparse row matrix.
All algorithms have been implemented through just-in-time compilation with numba.

## Features

Currently, BusFactorX offers the class *Project* that offers basic functionalieite to handle a bipartite graph of people and tasks and offers the following bus-factor approximation algorithms.

- **Maximum Redundant Set**: which defines the bus-factor in terms of *structurally redundant* people; that is, the people that can be safely removed from a project without harming it. This definition is due to Zazworka et al. (2010) and encodes their optimistic definition of bus-factor. In this context, a project is considered safe if at least $t\%$ of tasks are covered by at least one person, where $t$ is an input threshold.

- **Minimum Critical Set**: which defines the bus-factor in terms of *structurally critical* people; that is, the people that once removed from a project leave it in danger. This definition is due to Avelino et al. (2016) and is in relation with the pessimistic estimate from Zazworka et al. (2010). In this context, a project is considered in danger when the number of tasks covered by at least one person is lower than $t\%$.

- **Bus-Factor as Network Robustness**: The previous two approaches rely on thresholds to define when a project stalls and have also other weaknessess — see Piccolo et al. (2025) and Piccolo et al. (2024). As such, Piccolo et al. (2024, 2025) have proposed to compute the bus-factor of a project by evaluating the largest number of tasks connected into a single component, as people are removed from the project — a mechanism inspired by research on network robustness. The method builds the decay curve of the number of tasks as people are removed from the project and computes the area under the curve. This methods avoids thresholds and does not suffer from the weaknesses of prior methods. The area under the curve is normalized and can be interpreted as the fraction of people that have to be removed from the project for it to be considered in danger.

In order to offer efficient implementations of the aforementioned methods, BusFactorX also offers just-in-time compiled (through numba) implementations of a Union-Find data structure and a BitField data structure. In addition, an efficient set-cover approximation algorithm is used to implement the aproximation algorithm for the Maximum Redundant Set.

## Synthetic graphs

Finally, the package contains 1700 synthetic power-law graphs, with 7500 people and 10000 tasks, sampled at increasing levels of assortativity (degree correlation), a network measure correlated with network robustness. The graphs have been sampled, with an MCMC, from the statistical ensemble $\{G\}$ with probability measure $\mu(G) \propto e^{-H(G)}$, induced by the Hamiltonian $H(G) = −J \sum_{(i,j) \in E} k_i k_j$, where $J$ is a constant which regulates the level of degree correlation in the sampled graphs and $k_i$ indicates the degree of node $i$. The sampling procedure expores 17 equally spaced values of $J$ in the range $[-0.002, 0.002]$, corresponding to a degree correlation in between -0.6 and 0.6. For each value of $J$, 100 independent graphs have been sampled.

All graphs files are provided in the edgelist format and can be opened in networkx. Each file is named following the pattern "Net\_\$SAMPLENUM\$\_J\_$VAL$.edgelist", where $ \$ $SAMPLENUM$ \$ $ is the sample number and \$VAL\$ is the value of $J$. For example, "Net\_61\_J\_0.002.edgelist" refers to the 61st network sampled with $J = 0.002$.

## Usage

Here's a simple example of how to use BusFactorX:

Let B be a biadjacency scipy sparse row matrix representing a bipartite graph of people to tasks, the bus-factor of B can be computed with:

```python
from busfactorx import Project

# Example usage
P = Project(B)
bus_factor = P.compute_bus_factor_network_robustness()
print(bus_factor)
```

## Requirements

BusFactorX requires the following dependencies:

```txt
numba>=0.60.0
numpy>=1.26.0
scipy>=1.13.0
```

You can install them using:

```bash
pip install -r requirements.txt
```

## Citation

If you use BusFactorX in your research, please cite the following references:

```
@inproceedings{piccolo2024evaluating,
author = {Piccolo, Sebastiano A. and De Meo, Pasquale and Terracina, Giorgio},
title = {Evaluating and improving projects’ bus-factor: a network analytical framework},
year = {2024},
publisher = {Springer},
booktitle = {16th International Conference, ASONAM 2024, Rende, Italy, September 2–5, 2024, Proceedings, Part I},
location = {Rende, Italy},
series = {LNCS}
}
```

## License

BusFactorX is released under the [MIT License](https://mit-license.org).

## Contact

For any issues or inquiries, open an issue on GitHub or contact me.
