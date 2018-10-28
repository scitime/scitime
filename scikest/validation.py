from scikest.estimate import Estimator
from scikest.train import Trainer
from scikest.utils import LogMixin, get_path, config, timeout

import numpy as np
import importlib



class Validation(Estimator, Trainer, LogMixin):
    def __init__(self):
        super().__init__()
        
        
    def model_validation_generator(self):
    
        #1. Train a few datapoints
        results=self._generate_data()

        #2. For the same data estimate with the model
        estimated=[]
        for i in range(len(results[0])):

            n=results[0].iloc[i,3]
            p=results[0].iloc[i,4]
            X,y = np.random.rand(n,p),np.random.rand(n,)

            dic_params=dict(results[0].iloc[i,5:])

            sub_module = importlib.import_module(self.params['module'])
            model = getattr(sub_module, self.algo)(**dic_params)
            
            estimated.append(float(self._estimate(model,X,y)))

        #3. Create Dataframe with both estimates and actuals
        final_output = results[1]
        final_output['estimated']=estimated
        return (final_output)

    
    def model_validation_analysis(self,drop_rate=None):
        
        if drop_rate:
            self.drop_rate=drop_rate
    
        actual_estimates_df=self.model_validation_generator()

        #Compute avg etc..
        avg_weighted_error=np.dot(actual_estimates_df['output'],(actual_estimates_df['output']-actual_estimates_df['estimated']))/sum(actual_estimates_df['output'])
        return(actual_estimates_df,avg_weighted_error)

