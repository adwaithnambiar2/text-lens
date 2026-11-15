import unittest
from experiment import evaluate,load,predict
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.report,cls.bundle=evaluate(load('data/intents.jsonl'))
    def test_group_isolation(self):
        self.assertFalse(set(self.report['train_families']) & set(self.report['test_families']))
        self.assertEqual(self.report['train_size']+self.report['test_size'],240)
    def test_metrics_and_counts(self):
        for result in self.report['results'].values():
            self.assertTrue(0<=result['macro_f1']<=1)
            self.assertEqual(sum(map(sum,result['confusion_matrix'])),self.report['test_size'])
    def test_prediction_shape(self):
        for predictions in predict(self.bundle,['Please reset my password','Where is my package?']).values():
            self.assertEqual(len(predictions),2)
            self.assertTrue(set(predictions)<=set(self.report['labels']))
if __name__=='__main__': unittest.main()
