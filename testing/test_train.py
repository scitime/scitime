import unittest

from scikest.train import Trainer

class TestTrain(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.trainer = Trainer(drop_rate=0.99999, verbose=0, algo='RandomForestRegressor')

    def test_generate_data(self):
        inputs, outputs = self.trainer._generate_data()
        assert inputs.shape[0] > 0
        assert outputs.shape[0] > 0
        TestTrain.inputs = inputs
        TestTrain.outputs = outputs

    def test_model_fit(self):
        meta_algo = self.trainer.model_fit(generate_data=False, df=TestTrain.inputs, outputs=TestTrain.outputs)
        assert type(meta_algo).__name__ == 'RandomForestRegressor'

if __name__ == '__main__':
    unittest.main()