import unittest

import numpy as np

from scitime._model import Model


class TestTrain(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.rf_trainer_metarf = Model(drop_rate=1,
                                       verbose=3,
                                       algo='RandomForestRegressor',
                                       meta_algo='RF')

        self.svc_trainer_metarf = Model(drop_rate=1,
                                        verbose=3, algo='SVC', meta_algo='RF')

        self.km_trainer_metarf = Model(drop_rate=1,
                                       verbose=3, algo='KMeans',
                                       meta_algo='RF')

        self.rf_trainer_metann = Model(drop_rate=0.9999,
                                       verbose=3, algo='RandomForestRegressor',
                                       meta_algo='NN')

        self.svc_trainer_metann = Model(drop_rate=1,
                                        verbose=3, algo='SVC', meta_algo='NN')

        self.km_trainer_metann = Model(drop_rate=1,
                                       verbose=3, algo='KMeans',
                                       meta_algo='NN')

    def test_generate_data_regression_metarf(self):
        rf_inputs, rf_outputs, _ = self.rf_trainer_metarf._generate_data()

        TestTrain.rf_inputs = rf_inputs
        TestTrain.rf_outputs = rf_outputs

        self.rf_inputs = rf_inputs
        self.rf_outputs = rf_outputs

        assert rf_inputs.shape[0] > 0
        assert rf_outputs.shape[0] == rf_inputs.shape[0]

    def test_transform_data(self):
        X, y, _, _ = self.rf_trainer_metarf._transform_data(self.rf_inputs, self.rf_outputs)

        new_X_1, new_X_2 = self.rf_trainer_metarf._scale_data(X, X, False)

        assert type(X) == np.ndarray
        assert type(y) == np.ndarray
        assert y.shape[0] == X.shape[0]
        assert type(new_X_1) == np.ndarray
        assert type(new_X_2) == np.ndarray

    def test_generate_data_classification_metarf(self):
        svc_inputs, svc_outputs, _ = self.svc_trainer_metarf._generate_data()

        TestTrain.svc_inputs = svc_inputs
        TestTrain.svc_outputs = svc_outputs

        assert svc_inputs.shape[0] > 0
        assert svc_outputs.shape[0] == svc_inputs.shape[0]

    def test_generate_data_unsupervised_metarf(self):
        km_inputs, km_outputs, _ = self.km_trainer_metarf._generate_data()

        TestTrain.km_inputs = km_inputs
        TestTrain.km_outputs = km_outputs

        assert km_inputs.shape[0] > 0
        assert km_outputs.shape[0] == km_inputs.shape[0]

    def test_validate_data_regression_metarf(self):
        rf_val_inputs, rf_val_outputs, rf_val_estimated_outputs, \
            rf_val_avg_weighted_error = self.rf_trainer_metarf.model_validate()

        assert rf_val_inputs.shape[0] > 0
        assert rf_val_outputs.shape[0] == rf_val_inputs.shape[0]
        assert rf_val_estimated_outputs.shape[0] == rf_val_inputs.shape[0]
        assert type(rf_val_avg_weighted_error) == np.float64

    def test_validate_data_classification_metarf(self):
        svc_val_inputs, svc_val_outputs, svc_val_estimated_outputs, \
            svc_val_avg_weighted_error = self.svc_trainer_metarf.model_validate()

        assert svc_val_inputs.shape[0] > 0
        assert svc_val_outputs.shape[0] == svc_val_inputs.shape[0]
        assert svc_val_estimated_outputs.shape[0] == svc_val_inputs.shape[0]
        assert type(svc_val_avg_weighted_error) == np.float64

    def test_validate_data_unsupervised_metarf(self):
        km_val_inputs, km_val_outputs, km_val_estimated_outputs, \
            km_val_avg_weighted_error = self.km_trainer_metarf.model_validate()

        assert km_val_inputs.shape[0] > 0
        assert km_val_outputs.shape[0] == km_val_inputs.shape[0]
        assert km_val_estimated_outputs.shape[0] == km_val_inputs.shape[0]
        assert type(km_val_avg_weighted_error) == np.float64

    def test_model_fit_regression_metarf(self):
        rf_meta_algo = \
            self.rf_trainer_metarf.model_fit(generate_data=False,
                                             inputs=TestTrain.rf_inputs,
                                             outputs=TestTrain.rf_outputs)

        assert type(rf_meta_algo).__name__ == 'RandomForestRegressor'

    def test_model_fit_classification_metarf(self):
        svc_meta_algo = \
            self.svc_trainer_metarf.model_fit(generate_data=False,
                                              inputs=TestTrain.svc_inputs,
                                              outputs=TestTrain.svc_outputs)

        assert type(svc_meta_algo).__name__ == 'RandomForestRegressor'

    def test_model_fit_unsupervised_metarf(self):
        km_meta_algo = \
            self.km_trainer_metarf.model_fit(generate_data=False,
                                             inputs=TestTrain.km_inputs,
                                             outputs=TestTrain.km_outputs)

        assert type(km_meta_algo).__name__ == 'RandomForestRegressor'

    def test_generate_data_regression_metann(self):
        rf_inputs, rf_outputs, _ = self.rf_trainer_metann._generate_data()

        TestTrain.rf_inputs = rf_inputs
        TestTrain.rf_outputs = rf_outputs

        assert rf_inputs.shape[0] > 0
        assert rf_outputs.shape[0] == rf_inputs.shape[0]

    def test_generate_data_classification_metann(self):
        svc_inputs, svc_outputs, _ = self.svc_trainer_metann._generate_data()

        TestTrain.svc_inputs = svc_inputs
        TestTrain.svc_outputs = svc_outputs

        assert svc_inputs.shape[0] > 0
        assert svc_outputs.shape[0] == svc_inputs.shape[0]

    def test_generate_data_unsupervised_metann(self):
        km_inputs, km_outputs, _ = self.km_trainer_metann._generate_data()

        TestTrain.km_inputs = km_inputs
        TestTrain.km_outputs = km_outputs

        assert km_inputs.shape[0] > 0
        assert km_outputs.shape[0] == km_inputs.shape[0]

    def test_validate_data_regression_metann(self):
        rf_val_inputs, rf_val_outputs, rf_val_estimated_outputs, \
            rf_val_avg_weighted_error = self.rf_trainer_metann.model_validate()

        assert rf_val_inputs.shape[0] > 0
        assert rf_val_outputs.shape[0] == rf_val_inputs.shape[0]
        assert rf_val_estimated_outputs.shape[0] == rf_val_inputs.shape[0]
        assert type(rf_val_avg_weighted_error) == np.float64

    def test_validate_data_classification_metann(self):
        svc_val_inputs, svc_val_outputs, svc_val_estimated_outputs, \
            svc_val_avg_weighted_error = self.svc_trainer_metann.model_validate()

        assert svc_val_inputs.shape[0] > 0
        assert svc_val_outputs.shape[0] == svc_val_inputs.shape[0]
        assert svc_val_estimated_outputs.shape[0] == svc_val_inputs.shape[0]
        assert type(svc_val_avg_weighted_error) == np.float64

    def test_validate_data_unsupervised_metann(self):
        km_val_inputs, km_val_outputs, km_val_estimated_outputs, \
            km_val_avg_weighted_error = self.km_trainer_metann.model_validate()

        assert km_val_inputs.shape[0] > 0
        assert km_val_outputs.shape[0] == km_val_inputs.shape[0]
        assert km_val_estimated_outputs.shape[0] == km_val_inputs.shape[0]
        assert type(km_val_avg_weighted_error) == np.float64

    def test_model_fit_regression_metann(self):
        rf_meta_algo = \
            self.rf_trainer_metann.model_fit(generate_data=False,
                                             inputs=TestTrain.rf_inputs,
                                             outputs=TestTrain.rf_outputs)

        assert type(rf_meta_algo).__name__ == 'MLPRegressor'

    def test_model_fit_classification_metann(self):
        svc_meta_algo = \
            self.svc_trainer_metann.model_fit(generate_data=False,
                                              inputs=TestTrain.svc_inputs,
                                              outputs=TestTrain.svc_outputs)

        assert type(svc_meta_algo).__name__ == 'MLPRegressor'

    def test_model_fit_unsupervised_metann(self):
        km_meta_algo = \
            self.km_trainer_metann.model_fit(generate_data=False,
                                             inputs=TestTrain.km_inputs,
                                             outputs=TestTrain.km_outputs)

        assert type(km_meta_algo).__name__ == 'MLPRegressor'

    def test_random_search_metann(self):
        nn_meta_algo = \
            self.rf_trainer_metann._random_search(inputs=TestTrain.rf_inputs,
                                                  outputs=TestTrain.rf_outputs,
                                                  iterations=1)
        assert type(nn_meta_algo.best_estimator_).__name__ == 'MLPRegressor'


if __name__ == '__main__':
    unittest.main()
