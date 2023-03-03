import enum

EarlyStoppingMode = enum.Enum(
    "EarlyStoppingMode", ["BEST", "STOP", "PATIENCE"]
)


class EarlyStopping:
    def __init__(self, patience=10, maximize=True):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.maximize = maximize

    def __call__(self, score) -> EarlyStoppingMode:
        if self.best_score is None:
            self.best_score = score
            return EarlyStoppingMode.BEST

        if self.maximize:
            if score > self.best_score:
                self.best_score = score
                self.counter = 0
                return EarlyStoppingMode.BEST
        else:
            if score < self.best_score:
                self.best_score = score
                self.counter = 0
                return EarlyStoppingMode.BEST

        self.counter += 1
        if self.counter >= self.patience:
            return EarlyStoppingMode.STOP
        else:
            return EarlyStoppingMode.PATIENCE

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "counter": self.counter,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict["best_score"]
        self.counter = state_dict["counter"]
