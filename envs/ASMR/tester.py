from core.test.tester import Tester

class ASMRTester(Tester):
    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            for _ in episodes:
                self.history.add_evaluation_data({}, log=self.log_results)
