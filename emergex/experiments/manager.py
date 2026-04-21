############## Experiment Manager #############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

import jax.numpy as jnp

from ..core.simulate import (
    CRNSimulationRunner
)
from ..optimization.optimize import (
    CRNOptimizationFramework
)
from ..utils.helpers import ManagerClass
from .experiment import ExperimentGroup

###############################################

class OptimizeExperimentsManager(ManagerClass):
    def __init__(self, experimentGroups: list[ExperimentGroup]):
        self.ExperimentGroups = experimentGroups

        self.NormalizedWeights = jnp.array([exptGroup.Weight for exptGroup in self.ExperimentGroups])
        self.NormalizedWeights /= jnp.sum(self.NormalizedWeights)

    def getCostFn(self, optimizerFrameworkObj: CRNOptimizationFramework, simulationRunnerObj: CRNSimulationRunner):
        exptGroups = self.ExperimentGroups
        normWeights = self.NormalizedWeights
        
        precomputedTimeCourseData = []
        for experimentGroupObj in self.ExperimentGroups:
            timePointHandlerObj = experimentGroupObj.TimePointsHandler
            precomputedTimeCourseData.append([
                (
                    timePointHandlerObj.parseEvalPts(expt.TimeCourse),
                    jnp.arange(len(timePointHandlerObj.TotalEvalPts))
                ) for expt in experimentGroupObj.Experiments
            ])
        
        def getExperimentFitCost(y):

            exptCost = 0
            for i, experimentGroupObj in enumerate(exptGroups):
                crnInfo = experimentGroupObj.CRNInfo
                initCondInfo = optimizerFrameworkObj.getStartingConditions(y, crnInfo)
                
                for j, expt in enumerate(experimentGroupObj.Experiments):
                    results = simulationRunnerObj.runTimeCourse(precomputedTimeCourseData[i][j][0], crnInfo, initCondInfo)

                    signalCost = 0.0
                    for k, signal in enumerate(expt.Signals):
                        # resultOfInterest = signal.getNormalizeVMAP(initCondInfo.ComponentsDictionary)(
                        #     jnp.sum(
                        #         jnp.atleast_2d(
                        #             results[precomputedTimeCourseData[i][j][1], signal.getRelevantComponentIndices(crnInfo.CompNameList)]
                        #         ),
                        #         axis=0
                        #     )
                        # )
                        time_idx = precomputedTimeCourseData[i][j][1]
                        comp_idx = signal.getRelevantComponentIndices(crnInfo.CompNameList)                        
                        raw_result = results[time_idx][:, comp_idx]                        
                        summed_result = jnp.sum(raw_result, axis=-1)                        
                        resultOfInterest = signal.getNormalizeVMAP(initCondInfo.ComponentsDictionary)(summed_result)
                        
                        signalCost += jnp.sum(jnp.square(resultOfInterest - signal.DataPoints) * expt.SignalWeights[k])
                    
                    exptCost += signalCost * experimentGroupObj.ExperimentWeights[j] * normWeights[i]
            
            return exptCost
        
        return getExperimentFitCost
