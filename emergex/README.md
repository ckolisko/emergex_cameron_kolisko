# EmergeX

EmergeX is a Python library for building chemical reaction networks (CRNs), simulating their dynamics, and optimizing model parameters against either desired emergent behaviors or experimental time-course data. It is meant for workflows where you want to define a reaction system, specify what the system should do, and then search for parameter sets that make that behavior or fit emerge.

## Example outputs

The videos below are intended to be served from this package's static `assets/` folder rather than generated output directories.

### Small CRN parameter evolution

<video src="assets/smallCRN_parameters.mp4" controls muted playsinline width="720"></video>

### Small CRN behavior evolution

<video src="assets/smallCRN_behaviors.mp4" controls muted playsinline width="720"></video>

### Oscillator behavior evolution

<video src="assets/oscillateBehaviors.mp4" controls muted playsinline width="720"></video>

### Bistable genelet optimization

<video src="assets/bistableGeneletSwitch.mp4" controls muted playsinline width="720"></video>

## Installation

EmergeX targets Python 3.11+.

```bash
pip install .
```

For editable development installs:

```bash
pip install -e .
```

The project depends on `numpy`, `scipy`, `jax`, `diffrax`, `equinox`, `optax`, `dill`, `matplotlib`, `imageio`, and `imageio-ffmpeg`.

## What EmergeX provides

EmergeX is organized into a few main layers:

1. `emergex.crn`
   Build reaction networks from `Reaction`, `Component`, and `ReactionNetwork`.
2. `emergex.core`
   Define simulation metadata, time spans, interruptions, and evaluation schedules.
3. `emergex.behaviors`
   Express desired `HIGH` and `LOW` behaviors across time windows.
4. `emergex.experiments`
   Fit CRNs against measured experimental signals.
5. `emergex.optimization`
   Choose free and linked parameters, then run gradient-based optimization.
6. `emergex.visualization`
   Save landscape plots and MP4 animations of parameter changes and optimization progress.

## Basic workflow

The typical EmergeX workflow is:

1. Build a reaction network.
2. Define a simulation time course, including any staged additions or perturbations.
3. Specify either target behaviors or experimental data.
4. Select which parameters are allowed to change.
5. Run optimization.
6. Save the run and render visualizations.

## Building a CRN

At the lowest level, EmergeX works with reactions, components, and a reaction network container.

```python
from emergex import Reaction, Component, ReactionNetwork

rxn1 = Reaction(5e-5, "A + B -> C")
rxn2 = Reaction(1e-2, "C + D <-> CD", backwardRate=1.0)
rxn3 = Reaction(3e-2, "CD -> D + E")

network = ReactionNetwork()
network.addRxns([rxn1, rxn2, rxn3])
network.addComponents([
    Component("A", 100),
    Component("B", 80),
    Component("D", 5),
])
```

Important assumptions:

- Units are user-defined, but they must remain internally consistent.
- Stoichiometry is represented by repeating species names rather than using coefficients.
- The CRN can be simulated directly before optimization, which is useful for validating the baseline system.

## Direct simulation

You can simulate a reaction network without invoking the optimizer.

```python
network.simulateReactionFn(600, simDataResolution=201)
result = network.SimResults[-1]
```

This direct mode is the simplest way to inspect raw dynamics or test whether the network responds as expected.

## Time spans and interruptions

EmergeX supports staged simulations where reagents are added, removed, or replaced during the run.

```python
from emergex import TimeSpan, Interruption

interrupt = Interruption(network.Components["B"], 40, interruptionType="ADD")
time_course = [
    TimeSpan(3600),
    TimeSpan(3600, [interrupt]),
]
```

This is a core modeling feature because many CRN experiments involve sequential additions rather than a single uninterrupted trajectory.

## Optimizing for behaviors

Behavior optimization is the main design workflow used in the provided examples.

### Define target behaviors

Behaviors describe when a normalized signal should be high or low.

```python
from emergex import Behavior, BehaviorGroup, BehaviorTimeCourse

def normalize_by_D(x, concs):
    return x / concs["D"]

behavior_time_course = BehaviorTimeCourse(
    time_course,
    [
        BehaviorGroup(
            "CD",
            [
                Behavior("HIGH", 0.1 * 7200, 0.4 * 7200),
                Behavior("LOW", 0.5 * 7200, 1.0 * 7200),
            ],
            normalizeFn=normalize_by_D,
        )
    ],
)
```

The main objects are:

- `Behavior`: a target state over a time interval.
- `BehaviorGroup`: the species being scored plus the normalization rule.
- `BehaviorTimeCourse`: the time-course definition paired with one or more behavior groups.

### Define free and linked parameters

Optimization only changes parameters you explicitly expose.

```python
from emergex import FreeParameter, LinkedParameter

free_params = [
    FreeParameter(rxn1),
    FreeParameter(rxn2),
    FreeParameter(rxn3),
    FreeParameter(network.Components["B"]),
]
```

Use `LinkedParameter` when one quantity should be determined from another concentration or rate rather than optimized independently.

### Run optimization

```python
from emergex import (
    CRNInfoHandler,
    OptimizeBehaviorsManager,
    CRNSimulationRunner,
    CRNOptimizationFramework,
    CompiledOptimizationData,
)

manager = OptimizeBehaviorsManager(
    CRNInfoHandler(network),
    [behavior_time_course],
)

simulation_runner = CRNSimulationRunner()
optimizer = CRNOptimizationFramework(free_params)

result = CompiledOptimizationData(
    manager,
    simulation_runner,
    optimizer,
    iterationCount=100,
    callbackFrequency=4,
)
```

`CompiledOptimizationData` bundles together the objective definition, simulation runner, optimization framework, and the recorded optimization history.

## Saving and loading runs

Optimization runs can be serialized and reloaded later.

```python
result.save(DATA_STORE, "smallCRNOptResult")
loaded = CompiledOptimizationData.load(DATA_STORE / "smallCRNOptResult.pkl")
```

This is particularly useful because EmergeX preserves dynamic helper functions involved in normalization, interruptions, and linked-parameter logic as part of the saved object graph.

## Fitting experimental data

EmergeX also supports fitting parameters against measured signals instead of abstract behavior windows.

This workflow uses:

- `Signal` to define a measured observable and its normalization.
- `Experiment` to group one or more signals over a specific time course.
- `ExperimentGroup` to collect compatible experiments with shared evaluation times and CRN structure.
- `OptimizeExperimentsManager` to construct the optimization objective.

In this mode, the optimizer minimizes the difference between simulated normalized trajectories and the supplied data points.

## Visualization

EmergeX includes utilities for turning an optimization run into figures and videos.

```python
from emergex import (
    saveBehaviorLandscapeVisualization,
    saveFreeParametersVisualization,
    saveBehaviorResultsVisualization,
    saveExperimentResultsVisualization,
)
```

Typical usage:

```python
saveBehaviorLandscapeVisualization(
    manager,
    fileLocation=DATA_STORE,
    fileName="smallCRN_landscape",
    timeUnits="min",
)

saveFreeParametersVisualization(
    result,
    iterationList=[1, 5, 10, 20, 50, 100],
    fileLocation=DATA_STORE,
    fileName="smallCRN_parameters",
    fps=6,
)

saveBehaviorResultsVisualization(
    result,
    iterationList=[1, 5, 10, 20, 50, 100],
    fileLocation=DATA_STORE,
    fileName="smallCRN_behaviors",
    fps=6,
    timeUnits="min",
)
```

These functions produce the same kinds of videos shown at the top of this README:

- parameter evolution videos,
- behavior convergence videos,
- and experiment fit videos.

## Suggested starting points

For a quick entry into the library:

1. Start with the direct CRN example to understand `Reaction`, `Component`, and `ReactionNetwork`.
2. Move to the small behavior-optimization example to see the full end-to-end workflow.
3. Inspect the saved visual outputs to understand how optimization progress is reported.

## Citing EmergeX

If you use EmergeX in academic work, cite the software directly and include the exact version or commit used in your study.

Suggested citation:

Yancey, C., & Kolisko, C. (2026). *EmergeX* (Version 0.0.1) [Computer software]. GitHub. https://github.com/YanceyColin/EmergeX

Suggested BibTeX:

```bibtex
@software{yancey_kolisko_emergex_2026,
  author = {Yancey, Colin and Kolisko, Cameron},
  title = {EmergeX},
  year = {2026},
  version = {0.0.1},
  url = {https://github.com/YanceyColin/EmergeX},
  note = {Computer software}
}
```

If you publish from a development snapshot rather than a release, replace the version with the release tag or commit hash you actually used.
