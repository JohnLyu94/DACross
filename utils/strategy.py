class TwoSideEarlyStopping:
    def __init__(self, patience):
        self.count = 0
        self.patience = patience

    def __call__(self, train, val):
        if val > train:
            self.count += 1
        else:
            self.count = 0

    def judge(self):
        if self.count >= self.patience:
            return True
        else:
            return False


class NonImprovementEarlyStopping:
    def __init__(self, delta, patience):
        self.count = 0
        self.delta = delta
        self.patience = patience
        self.min_error = float("inf")

    def __call__(self, current_error):
        if current_error < self.min_error:
            self.min_error = current_error
        generalization_loss = 100 * ((current_error / self.min_error) - 1)

        if generalization_loss > self.min_error:
            self.count += 1
        else:
            self.count = 0

    def judge(self):
        if self.count >= self.patience:
            return True
        else:
            return False
