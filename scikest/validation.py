from scikest.estimate import Estimator
from scikest.train import Trainer
from scikest.utils import LogMixin, get_path, config, timeout

import numpy as np
import importlib



class Validation(Trainer, LogMixin):
    def __init__(self):
        super().__init__()
        
            
    def model_validation_analysis(self,drop_rate=None):
        
        if drop_rate:
            self.drop_rate=drop_rate

        self.validation = True    
        actual_estimates_df=self._generate_data()

        #Compute avg etc..
        avg_weighted_error=np.dot(actual_estimates_df[1]['output'],(actual_estimates_df[1]['output']-actual_estimates_df[2]['estimated_outputs']))/sum(actual_estimates_df[1]['output'])
        return(actual_estimates_df,avg_weighted_error)
