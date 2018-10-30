from scikest.estimate import Estimator
from scikest.train import Trainer
from scikest.utils import LogMixin, get_path, config, timeout

import numpy as np


class Validation(Estimator, Trainer, LogMixin):
    def __init__(self):
        super().__init__()
    
    def model_validation_analysis(self,drop_rate=None):
        if drop_rate:
            self.drop_rate=drop_rate
    
        actual_estimates_df=self._generate_data(self._estimate)
        actual=actual_estimates_df[1]["output"]
        estimates=actual_estimates_df[2]["estimated_time"]
        
        #Compute avg etc..
        avg_weighted_error=np.dot(actual,(actual-estimates))/sum(actual)

        final_output = actual_estimates_df[1]
        final_output['estimated']=actual_estimates_df[2]
        return(final_output,avg_weighted_error)
        #return(actual_estimates_df)

