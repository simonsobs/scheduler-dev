# scheduler

## Related Packages

* [**scheduler**](https://github.com/simonsobs/scheduler) - This package. The
  core scheduling library, a.k.a. `schedlib`.
* [**scheduler-server**](https://github.com/simonsobs/scheduler-server) - The
  Flask API for fetching schedules.
* [**scheduler-web**](https://github.com/simonsobs/scheduler-web) - The web
  front end for the scheduler.

## Installation
First clone this repository, and then install with pip:
```bash
git@github.com:simonsobs/scheduler.git
pip install .
```

Alternatively, pip install directly from GitHub:
```bash
pip install schedlib@git+https://github.com/simonsobs/scheduler.git
```
## Basic design
The basic task of scheduler is to take sequences of planned observations, optimize them based on observational constraint set by observer and instrument, and convert them into a series of commands to operate the telescope. To achieve this goal, the scheduler divides this task into three steps: 
1. initialize sequences of timed blocks (`Block`) which represent planned observations;
2. apply transformations (`Rule`) to the sequences of blocks to optimize the sequence based on given constraints. An example transformation could be `sun-avoidance` which removes blocks that are too close to the sun;
3. convert the sequence of blocks into a sequence of commands to operate the telescope.

A `Policy` defines how this three-step process is orchastrated. In other words, a policy defines how sequences of `Blocks` get transformed into commands. Step 1 is implemented with `Policy.init_seq`, step 2 is implemented with `Policy.apply`, and step 3 is implemented with `Policy.seq2cmd`. Different policies can be implemented to correspond to the needs of different instruments. 

## Usage
[Confluence link for instructions on setting up and using scheduler](https://simonsobs.atlassian.net/wiki/spaces/SOPS/pages/289374233/Using+Scheduler)
