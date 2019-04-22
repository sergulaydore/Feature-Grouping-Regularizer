import torch
import random
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.utils import gen_batches, shuffle, check_random_state
from rena import ReNA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import set_torch_seed
import time


class Solver(object):
    def __init__(self, net, learning_rate, n_epochs,
                 batch_size, seed, n_phi=None, n_sample_rena=None,
                 masker=None, lambda_l2=0, lambda_l1=0,
                 early_stopping=False, patience=10, display=50
                ):

        self.net = net
        self.n_clusters = net.n_cluster if hasattr(net, 'n_cluster') else None
        self.params = list(net.parameters())
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = check_random_state(seed)
        random.seed(seed)
        self.seed = seed
        self.n_phi = n_phi
        self.n_sample_rena = n_sample_rena
        self.masker = masker
        self.loss = []
        self.test_loss = []
        self.valid_loss = []
        self.train_loss = []
        self.lambda_l2 = lambda_l2 # L2 regularizer
        self.lambda_l1 = lambda_l1  # L1 regularizer
        self.time_track = [0]
        self.test_accuracy_hist = [0]
        self.train_accuracy_hist = [0]
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_loss = np.inf
        self.no_improvement = 0
        self.display = display

    def precompute_clusters(self, X_train, n_phi, n_samples,
                            n_sample_rena, n_clusters,
                            masker):
        clusters = []
        X_train = X_train.data.numpy() if not type(X_train) == np.ndarray else X_train

        for idx_phi in range(n_phi):
            random_indices = random.sample(range(0, n_samples),
                                           n_sample_rena)
            cluster = ReNA(scaling=True,
                           n_clusters=n_clusters,
                           masker=masker)
            cluster.fit(X_train[random_indices, :])
            clusters.append(cluster)

        return clusters


    def batch_update(self, X_batch, y_batch, cluster):

        input_ = torch.from_numpy(X_batch).float() if type(X_batch) == np.ndarray else X_batch
        target = torch.from_numpy(y_batch) if type(y_batch) == np.ndarray else y_batch

        out = self.net.forward(input_, cluster, training=True)

        l2_reg = autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        l1_reg = autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        for idx, p in enumerate(self.params):
            if self.net.n_cluster is not None and idx <= 1:
                if idx == 0:
                    l2_reg = l2_reg + p.norm(2)
                    l1_reg = l1_reg + p.norm(1)
                elif idx == 1:
                    pass
            else:
                l2_reg = l2_reg + p.norm(2)
                l1_reg = l1_reg + p.norm(1)
        loss = self.criterion(out, target.long()) + self.lambda_l2 * l2_reg + self.lambda_l1 * l1_reg
        self.net.zero_grad()
        loss.backward()

        # manual sgd
        for idx, p in enumerate(self.params):
            if self.net.n_cluster is not None and idx <= 1:
                if idx == 0:
                    grad = cluster.inverse_transform(self.params[1].grad.data.numpy())
                    p.data.sub_(torch.from_numpy(grad).float() * self.learning_rate)
                    self.params[1].grad.zero_()
                elif idx == 1:
                    pass
            else:
                p.data.sub_(p.grad.data * self.learning_rate)
                p.grad.zero_()

        batch_loss = self.criterion(out, target.long())

        return float(batch_loss.data)

    def train_numpy(self, X_train, y_train, X_test, y_test):

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                            test_size=0.33,
                                                            random_state=self.seed)
        n_samples = X_train.shape[0]

        if hasattr(self.net, 'n_cluster') and self.net.n_cluster:
            clusters = self.precompute_clusters(X_train, self.n_phi, n_samples,
                                                self.n_sample_rena, self.n_clusters,
                                                self.masker)
        else:
            clusters = None
        if self.batch_size is None:
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        batch_size_test = 5 # this is just to fit the CNNs to memory

        start_training_epoch = time.time() 

        for idx in range(self.n_epochs):
            if idx % self.display == 0:
                print("Epoch", idx)
            X_train, y_train = shuffle(X_train, y_train, random_state=self.random_state)
            accumulated_loss = 0.0
            for batch_slice in gen_batches(n_samples, batch_size):
                if clusters is not None:
                    idx_cluster = random.randint(0, self.n_phi - 1)
                    cluster = clusters[idx_cluster]
                else:
                    cluster = None
                X_batch, y_batch = X_train[batch_slice, :], y_train[batch_slice]
                batch_loss = self.batch_update(X_batch, y_batch, cluster)
                accumulated_loss += batch_loss * (batch_slice.stop -
                                                  batch_slice.start)

            end_training_epoch = time.time()
            epoch_training = end_training_epoch - start_training_epoch
            self.time_track.append(self.time_track[-1] + epoch_training)

            iteration_loss = accumulated_loss / X_train.shape[0]
            self.loss.append(iteration_loss)

            pred, accuracy_test, iteration_test_loss = self.predict_numpy(X_test, y_test, batch_size_test)
            self.test_loss.append(iteration_test_loss)
            self.test_accuracy_hist.append(accuracy_test)

            _, _, iteration_valid_loss = self.predict_numpy(X_valid, y_valid, batch_size_test)
            self.valid_loss.append(iteration_valid_loss)

            _, accuracy_train, iteration_train_loss = self.predict_numpy(X_train, y_train, batch_size_test)
            self.train_loss.append(iteration_train_loss)
            self.train_accuracy_hist.append(accuracy_train)

            if idx % self.display == 0:
                print("Accuracy in Percentage for train: %.2f, for test : %.2f" % (100*accuracy_train, 100*accuracy_test))

            if self.early_stopping:
                if iteration_valid_loss > self.best_loss + 0.01:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0

            self.best_loss = min(self.best_loss, iteration_valid_loss)

            if self.early_stopping and self.no_improvement > self.patience:
                print("Quitting training for early stopping at epoch ", idx)
                print("Accuracy train: %.2f, test : %.2f" % (accuracy_train, accuracy_test))
                break

    def predict_numpy(self, X, y, batch_size):
        if batch_size is None:
            input_ = torch.from_numpy(X).float() if type(X) == np.ndarray else X
            target = torch.from_numpy(y) if type(y) == np.ndarray else y
            out = self.net.forward(input_, cluster=None, training=False)
            pred = out.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(target)
            iteration_test_loss = self.criterion(out, target)
        else:
            # this is needed for CNNs because they require more memory so we need
            # to feed test data as batches
            n_samples = X.shape[0]
            weighted_loss = weighted_correct = 0.0
            for batch_slice in gen_batches(n_samples, batch_size):
                X_batch, y_batch = X[batch_slice, :], y[batch_slice]
                input_ = torch.from_numpy(X_batch).float() if type(X_batch) == np.ndarray else X_batch
                target = torch.from_numpy(y_batch) if type(y_batch) == np.ndarray else y_batch
                out = self.net.forward(input_, cluster=None, training=False)
                pred = out.max(1, keepdim=True)[1] # TODO: pred here is just the last batch
                correct = pred.eq(target.view_as(pred)).sum().item()
                weighted_correct += correct 
                weighted_loss += self.criterion(out, target) * (batch_slice.stop - batch_slice.start)
            accuracy = weighted_correct / n_samples
            iteration_test_loss = weighted_loss / n_samples
        return pred, accuracy, float(iteration_test_loss.data)
