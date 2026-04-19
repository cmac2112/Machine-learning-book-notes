class DataProcessor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def return_split_data(self):
        #the given dataset is already split into a training set (first 60,000) and a test set last (10,000)

        self.train_x, self.test_x = self.x[:60000], self.x[60000:]
        self.train_y, self.test_y = self.y[:60000], self.y[60000:]

        return self

    #returns true false values for the given label to create target vectors
    def return_values_where(self, val: str):
        y_train_vals = (self.train_y == val)
        y_test_vals = (self.test_y == val)

        return y_train_vals, y_test_vals
    def return_values(self):
        return self.train_x, self.train_y, self.test_x, self.test_y