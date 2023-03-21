import os
import re
import shutil


class DirManager:
    def __init__(self, root_dir, maximize=False):
        self.root_dir = root_dir
        self.maximize = maximize

        self.best_pattern = r"model_best.*_(\d*\.?\d*)"
        self.step_pattern = r"model_(\d+)"

        os.makedirs(root_dir, exist_ok=True)

    @property
    def best_model(self):
        best_score = None
        best_model = None
        for filename in os.listdir(self.root_dir):
            match = re.match(self.best_pattern, filename)
            if match:
                score = float(match.group(1))
                if (
                    best_score is None
                    or (self.maximize and score > best_score)
                    or (not self.maximize and score < best_score)
                ):
                    best_score = score
                    best_model = os.path.join(self.root_dir, filename)
        return best_model

    @property
    def last_model(self):
        last_model = os.path.join(self.root_dir, "model_last")
        if os.path.exists(last_model):
            return last_model
        last_step = None
        last_model = None
        for filename in os.listdir(self.root_dir):
            match = re.match(self.step_pattern, filename)
            if match:
                step = int(match.group(1))
                if last_step is None or step > last_step:
                    last_step = step
                    last_model = os.path.join(self.root_dir, filename)
        return last_model

    def purge_old(self, k=None):
        """
        Keeps only the last k models.
        If k is None, keeps all models.
        """
        if k is None:
            return
        steps = []
        for filename in os.listdir(self.root_dir):
            match = re.match(self.step_pattern, filename)
            if match:
                steps.append(int(match.group(1)))
        steps.sort()
        purge_list = steps[:-k]
        for filename in os.listdir(self.root_dir):
            match = re.match(self.step_pattern, filename)
            if match and int(match.group(1)) in purge_list:
                shutil.rmtree(os.path.join(self.root_dir, filename))

    def purge_worse(self, k=None):
        """
        Keeps only the best k models.
        If k is None, keeps all models.
        """
        if k is None:
            return
        scores = []
        for filename in os.listdir(self.root_dir):
            match = re.match(self.best_pattern, filename)
            if match:
                scores.append(float(match.group(1)))
        scores.sort(reverse=self.maximize)
        purge_list = scores[k:]
        for filename in os.listdir(self.root_dir):
            match = re.match(self.best_pattern, filename)
            if match and float(match.group(1)) in purge_list:
                shutil.rmtree(os.path.join(self.root_dir, filename))
