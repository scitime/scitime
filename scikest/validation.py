from scikest.estimate import Estimator
from scikest.train import Trainer
from scikest.utils import LogMixin, get_path, config, timeout

import numpy as np
import pandas as pd
import importlib
import itertools


class Validation(Estimator, Trainer, LogMixin):
    def __init__(self):
        super().__init__()

    def _model_validation_generator(self,X,y,dic_params):
        sub_module = importlib.import_module(self.params['module'])
        model = getattr(sub_module, self.algo)(**dic_params)
        estimated=float(self._estimate(model,X,y))
        return estimated        
        
    def _generate_data_for_validation(self):
            """
            measures training runtimes for a set of distinct parameters
            saves results in a csv (row by row)

            :return: inputs, outputs
            :rtype: pd.DataFrame
            """
            if self.verbose >= 2:
                self.logger.info('Generating dummy training durations to create a training set')
            inputs = []
            outputs = []
            estimated_time=[]
            parameters_list = list(self.params['internal_params'].keys())
            external_parameters_list = list(self.params['external_params'].keys())
            concat_dic = dict(**self.params['external_params'], **self.params['internal_params'])

            algo_type = self.params["type"]
                
            # in this for loop, we fit the estimated algo multiple times for random parameters and random input (and output if the estimated algo is supervised)
            # we use a drop rate to randomize the parameters that we use

            for permutation in itertools.product(*concat_dic.values()):
                n, p = permutation[0], permutation[1]
                if algo_type == "classification":
                    num_cat = permutation[2]
                    parameters_dic = dict(zip(parameters_list, permutation[3:]))
                else:
                    parameters_dic = dict(zip(parameters_list, permutation[2:]))

                # computing only for (1-self.drop_rate) % of the data
                random_value = np.random.uniform()
                if random_value > self.drop_rate:
                    final_params = dict(zip(external_parameters_list + parameters_list, permutation))
                    # handling max_features > p case
                    try:
                        row_input = [self.memory.total, self.memory.available, self.num_cpu] + [i for i in permutation]

                        # fitting the models
                        if algo_type == "classification":
                            time_measured = self._measure_time(n, p, parameters_dic, num_cat, forValidation=True)
                            row_output=time_measured[0]
                            X=time_measured[1]
                            y=time_measured[2]
                            estimated_time.append(self._model_validation_generator(X,y,parameters_dic))   
                        else:
                            time_measured = self._measure_time(n, p, parameters_dic, forValidation=True)
                            row_output=time_measured[0]
                            X=time_measured[1]
                            y=time_measured[2]
                            estimated_time.append(self._model_validation_generator(X,y,parameters_dic))  

                        outputs.append(row_output)
                        inputs.append(row_input)  

                    except Exception as e:
                        if self.verbose >= 1:
                            self.logger.warning(f'model fit for {final_params} throws an error')

            inputs = pd.DataFrame(inputs, columns=self.params['other_params'] + external_parameters_list + parameters_list)
            outputs = pd.DataFrame(outputs, columns=['output'])
            estimated_time = pd.DataFrame(estimated_time , columns=['estimated_time'])
            return inputs, outputs, estimated_time
    
    def model_validation_analysis(self,drop_rate=None):
        if drop_rate:
            self.drop_rate=drop_rate
    
        actual_estimates_df=self._generate_data_for_validation()
        actual=actual_estimates_df[1]["output"]
        estimates=actual_estimates_df[2]["estimated_time"]
        
        #Compute avg etc..
        avg_weighted_error=np.dot(actual,(actual-estimates))/sum(actual)

        final_output = actual_estimates_df[1]
        final_output['estimated']=actual_estimates_df[2]
        return(final_output,avg_weighted_error)
        #return(actual_estimates_df)

