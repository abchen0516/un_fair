import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = self.output_layer(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = torch.softmax(self.linear(x), dim=-1)
        return self.linear(x)


class NaiveBayesClassifier:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_means_ = None
        self.feature_vars_ = None

    def fit(self, data_loader):
        X, y = [], []
        for batch in data_loader:
            tensorX, tensory = batch[0], batch[1]
            X.append(tensorX)
            y.append(tensory)

        X, y = torch.cat(X), torch.cat(y)

        classes = torch.unique(y)
        self.class_log_prior_ = torch.zeros(len(classes), dtype=torch.float)
        self.feature_means_ = torch.zeros((len(classes), X.size(1)), dtype=torch.float)
        self.feature_vars_ = torch.zeros((len(classes), X.size(1)), dtype=torch.float)

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = torch.log(torch.tensor(X_c.size(0) / X.size(0)))
            self.feature_means_[i, :] = X_c.mean(dim=0)
            self.feature_vars_[i, :] = X_c.var(dim=0)

    def predict(self, data_loader):
        X, y = [], []
        for batch in data_loader:
            tensorX, tensory = batch[0], batch[1]
            X.append(tensorX)
            y.append(tensory)

        X, y = torch.cat(X), torch.cat(y)

        log_probs = torch.zeros((X.size(0), len(self.class_log_prior_)))

        for i in range(len(self.class_log_prior_)):
            # Using the Gaussian probability density function log form
            mean = self.feature_means_[i]
            var = self.feature_vars_[i]
            exponent = torch.exp(-((X - mean) ** 2) / (2 * var))
            log_prob = -0.5 * torch.log(2 * torch.pi * var) - ((X - mean) ** 2) / (2 * var)
            log_probs[:, i] = log_prob.sum(1) + self.class_log_prior_[i]

        return log_probs.argmax(dim=1), log_probs
