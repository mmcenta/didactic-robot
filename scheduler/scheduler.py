
from math import log


class AveragedDuration:
    def __init__(self, beta):
        self.beta = beta
        self.corrected_average = 0.0
        self.average = 0.0
        self.num_updates = 0

    def update(self, duration):
        # Use the exponentialy weighted average
        self.average = self.beta * self.average + (1 - self.beta) * duration
        self.num_updates += 1

        if self.num_updates < 1 / (1 - self.beta):
            # If not yet warmed up, correct bias
            self.corrected_average = self.average / (1 - self.beta ** self.num_updates)
        else:
            # If warmed up, just use the normal value
            self.corrected_average = self.average

    def get_average(self):
        return self.corrected_average


class Scheduler:
    def __init__(self, time_budget=2400, time_step=60, beta=0.9):
        # Set the Scheduler parameters
        self.time_budget = time_budget
        self.time_step = time_step
        self.beta = beta
        
        # Initilize the attributes used to record the time of updates and decisions 
        self.last_decision_time = 0.0
        self.last_decision_accuracy = 0.0
        
        # Initialize the attributes used to calculate the gain 
        self.running_accuracy = 0.0
        self.test_duration = AveragedDuration()

    def _time_transform(self, t):
        return (log(1 + t/self.time_step) / log(1 + t/self.time_budget))

    def _calculate_gain(self, time, accuracy):
        return (1 - self._time_transform(time + self.test_duration.get_average())) * (accuracy - self.running_accuracy)

    def update_test(self, duration):
        # Update the average test duration
        self.test_duration.update(duration)

    def should_test(self, time, accuracy):
        # Calculate gains
        current_gain = self._calculate_gain(time, accuracy)
        last_decision_gain = self._calculate_gain(self.last_decision_time, self.last_decision_accuracy)

        # Update attributes
        self.last_decision_time = time
        self.last_decision_accuracy = accuracy

        # Make decision
        if last_decision_gain > current_gain:
            self.running_accuracy = accuracy
            return True
        else:
            return False