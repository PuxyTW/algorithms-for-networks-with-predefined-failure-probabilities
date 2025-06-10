# algorithms-for-networks-with-predefined-failure-probabilities

This repository contains the source code for the thesis "Arborescence Routing in networks with predefined failure probabilities" by the TU Dortmund student Andrew Carneiro Zen. The framework was 
based on the work provided in [DSN 2019: Bonsai: Efficient Fast Failover Routing Using Small Arborescences](https://ieeexplore.ieee.org/document/8809517) and their source code was based on the creation from Ilya Nikolaevskiy, Aalto University, Finland(https://ieeexplore.ieee.org/document/7728092). 

This thesis was supervised by the leading routing professor in the technical university of Dortmund,  [Klaus-Tycho Foerster](https://ktfoerster.github.io/)

## Overview

* benchmark_graphs: directory to be filled with network topologies used in the experiments
* results: directory to which csv and other output files are written

* arborescence.py: arborescence decomposition and helper algorithms
* routing_stats.py: routing algorithms, simulation and statistic framework
* objective_function_experiments.py: objective functions, independence and SRLG experiments
* srds2019_experiments.py: experiments for SRDS 2019 / TSDC 2022 paper
* dsn2019_experiments.py: experiments for DSN 2019 paper
* infocom2019_experiments.py: experiments for INFOCOM 2019 paper
* infocom2021_experiments.py: experiments for INFOCOM 2021 paper
* benchmark_template.py: template to compare algorithms

To e.g. run the experiments for the SRDS paper, execute the corresponding python file:
```
python srds2019_experiments.py
```
With additional arguments the experiments can be customised (see main function of the python file). E.g., 
In case of questions, feel free to message me on (andrew.carneiro@tu-dortmund.de)