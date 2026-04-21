############## Experiment Compiler ############
#                                             #
#              By: Cameron Kolisko            #
#              Created March 2026             #
#            Last Edited: 03/10/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

# Class where, given results reader and crn data, combines to make experiment group. Uses ExperimentMaker.

from typing import Optional
import jax.numpy as jnp

from .experiment import(
    Signal,
    Experiment,
    ExperimentGroup,
    EvaluationPointsHandler
)

from ..core.timeCourse import(
    TimeSpan,
    Interruption
)
from ..crn.crnConstructor import(
    ReactionNetwork
)


class ExperimentCompiler:
    
    # Initialize the experiment with the:
    # List of time series, 
    # weight to indicate optimization weighting
    def __init__(self, timeSpans:list[TimeSpan], weight:float = 1.):
        
        self.Signals:list[Signal] = []
        self.Weight = weight
        self.TimeSpans:list[TimeSpan] = timeSpans

    # Adds a signal based on a list of unnormalized well data, potentially from an RR.
    # Normalizes data from 0-1.
    # low value - "0 value" of the data
    # high value = "1 value" of the data
    # well - a list of well data
    # fluConcInverse - Bool indicating if relation between fluorescence and concentration is inverted. Maybe capture with max signal conc. 
    def addSignal(self, signal):
        # Finally, throw all of it into a signal
        self.Signals.append(signal)
        
        
    # The final step, after adding the signals and inturruptions to timespans, is to make the Experiment proper.
    def convertToExperiment(self) -> Experiment:
        return Experiment(self.TimeSpans, self.Signals, self.Weight)


class ExperimentGroupCompiler:
    # For initialization, timeSeriesData is your times series for this experiment group.
    # Important, as need same number of time data points in all experiments in the group
    def __init__(self, timeData:list[jnp.ndarray], rn: ReactionNetwork, weight:float = 1, shiftTimeToZero: bool = True):
        self.TimeData:list[jnp.ndarray] = self.__shiftTimeDataToZero(timeData) if shiftTimeToZero else timeData
        self.Crn = rn
        self.ExpList:list[ExperimentCompiler] = []
        self.Weight = weight

    @staticmethod
    def __shiftTimeDataToZero(timeData: list[jnp.ndarray]) -> list[jnp.ndarray]:
        if len(timeData) == 0 or timeData[0].size == 0:
            return timeData

        startTime = float(timeData[0][0])
        shiftedTimeData = []
        for arr in timeData:
            shiftedTimeData.append(arr - startTime)
        return shiftedTimeData

    @staticmethod    
    def __getStartEndTime(well:list[jnp.ndarray], startBound:int|None, endBound:int|None) -> tuple[int, int]:
        if startBound is None:
            startBound = float(well[0][0])
        if endBound is None:
            endBound = float(well[-1][-1]) + 1
        return (startBound, endBound)

    @staticmethod    
    def __getStartEndIndex(well:list[jnp.ndarray], startBound:int|None, endBound:int|None) -> tuple[int, int]:
        if startBound is None:
            startBound = 0
        if endBound is None:
            endBound = len(well) - 1
        return (startBound, endBound)

    # based on provided exp group time data, will give you a nice time span all packaged up.
    def createTimeSpanByBreakInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, interruptions: Optional[list[Interruption]] = None):
        # Just find difference of times from start index and end index.
        (startIndex, endIndex) = self.__getStartEndIndex(self.TimeData, startBound, endBound)
        timeDiff = float(self.TimeData[endIndex][-1]) - float(self.TimeData[startIndex][0])
        return TimeSpan(timeDiff, interruptions)
    
 # Finds the closest start time at or after the given time
    def __find_closest_start(self, startTime):
        for span in self.TimeData:
            for time in span:
                if time >= startTime:
                    return float(time)
        raise Exception("No valid start time found given a start time of " + str(startTime))

    # Finds the closest end time before the given time
    def __find_closest_end(self, endTime):
        for span_idx in range(len(self.TimeData) - 1, -1, -1):
            span = self.TimeData[span_idx]
            for i in range(len(span) - 1, -1, -1):
                if span[i] < endTime:
                    return float(span[i])
        raise Exception("No valid end time found given a end time of " + str(endTime))


# Creates a largest time span that falls within the bounds for the given time data.
    def createTimeSpanByTimeInterval(self, startBound: Optional[int | float] = None, endBound: Optional[int | float] = None, interruptions: Optional[list[Interruption]] = None):
        (startTime, endTime) = self.__getStartEndTime(self.TimeData, startBound, endBound)
        # Now, find the times closest to the end and start.
        startTime = self.__find_closest_start(startTime)
        endTime = self.__find_closest_end(endTime)
        timeDiff = endTime - startTime 
        return TimeSpan(timeDiff, interruptions)

    def addExperiment(self, timeSpans:list[TimeSpan], weight = 1.0 ) -> ExperimentCompiler:
        exp = ExperimentCompiler(timeSpans, weight)
        self.ExpList.append(exp)
        return exp

    @staticmethod
    def __timeToJax(well:list[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate(well)

    def convertToExperimentGroup(self):
        
        handler = EvaluationPointsHandler(self.__timeToJax(self.TimeData))    
        listOfExperiments = []
        for i in self.ExpList:
            listOfExperiments.append(i.convertToExperiment())
        return ExperimentGroup(handler, self.Crn, listOfExperiments, self.Weight)

