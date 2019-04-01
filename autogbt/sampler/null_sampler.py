class NullTrainDataSampler:

    def sample(self, X, y, train_idx, valid_idx):
        train_X = X.loc[train_idx].reset_index(drop=True)
        valid_X = X.loc[valid_idx].reset_index(drop=True)
        train_y = y.loc[train_idx].reset_index(drop=True)
        valid_y = y.loc[valid_idx].reset_index(drop=True)
        return train_X, train_y, valid_X, valid_y
