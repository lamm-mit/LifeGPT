# conway_lib/optimizer.py

from skopt import gp_minimize
from skopt.space import Real, Integer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from .game import ConwayGame
from .tokenizer import Tokenizer
from .model import ConwayModel
import time
import random

class RegressionDataset(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    
    def __getitem__(self, index):
        return self.X_data[index]
    
    def __len__(self):
        return len(self.X_data)

def cycle(loader):
    while True:
        for data in loader:
            yield data

class Optimizer:
    def __init__(self, conway_game, conway_model, tokenizer, max_length, device, max_epochs=10, validate_every=10):
        self.conway_game = conway_game
        self.conway_model = conway_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.max_epochs = max_epochs
        self.validate_every = validate_every
        self.validation_data = self.conway_game.generate_validation_sets(A=100, N=32, I=2)
        self.X_data_val = ["@PredictNextState<" + state[0] + "> [" + state[1] + "]$" for state in self.validation_data]
        self.tokenized_val_data = self.tokenizer.texts_to_sequences(self.X_data_val, do_padding=True)

    def count_mismatches(self, ground_truth, pred):
        mismatches = sum(1 for gt, p in zip(ground_truth, pred) if gt != p)
        accuracy = 1 - mismatches / len(ground_truth)
        return mismatches, accuracy

    def extract_sample(self, string_input, start_token='[', end_token=']'):
        i = string_input.find(start_token)
        j = string_input.find(end_token)
        return string_input[i+1:j]

    def extract_task(self, string_input, end_task_token='>'):
        j = string_input.find(end_task_token)
        return string_input[:j+1]

    def generate_sample(self):
        self.conway_model.model.eval()
        inp_str = self.extract_task(random.choice(self.X_data_val))
        inp = torch.Tensor(self.tokenizer.texts_to_sequences([inp_str], do_padding=False)).to(self.device)
        inp = inp.transpose(0, 1)
        inp = inp.long()

        sample = self.conway_model.model.generate(
            prompts=inp,
            seq_len=self.max_length - inp.size(0),  # Generate tokens until max_length
            eos_token=self.tokenizer.end_token,
            cache_kv=True
        )
        
        output_str = self.tokenizer.sequences_to_texts(sample[:1])
        return output_str[0]

    def evaluate_accuracy(self):
        accuracy_list = []
        for _ in range(10):  # Evaluate on 10 example games
            sample_output = self.generate_sample()
            pred = self.extract_sample(sample_output)
            gt = self.extract_sample(random.choice(self.X_data_val))
            _, acc = self.count_mismatches(gt, pred)
            accuracy_list.append(acc)
        
        avg_accuracy = np.mean(accuracy_list)
        return avg_accuracy

    def objective(self, params):
        dim, depth, heads, attn_dim_head, train_size, batch_size, order_mean, order_std = params

        self.conway_game.order_mean = order_mean
        self.conway_game.order_std = order_std

        training_data = self.conway_game.generate_training_sets(A=train_size, N=32, I=2)
        X_data_train = ["@PredictNextState<" + state[0] + "> [" + state[1] + "]$" for state in training_data]

        tokenized_train_data = self.tokenizer.texts_to_sequences(X_data_train, do_padding=True)

        model_name = "Conway_Tuned"
        self.conway_model = ConwayModel(self.max_length, 256, self.device, model_name, dim=dim, depth=depth, heads=heads, attn_dim_head=attn_dim_head)
        print("model intitialized")

        train_dataset = RegressionDataset(tokenized_train_data)
        batch_size = int(batch_size)  # Ensure batch_size is an integer
        train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size, drop_last=True))
        print("train loader initialized")

        optimizer = torch.optim.Adam(self.conway_model.model.parameters(), lr=1e-4)
        total_loss = 0
        accuracies = []
        num_batches = len(tokenized_train_data) // batch_size

        # Set up metrics storage
        metrics = []

        # Set up plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Batch")
        ax1.set_ylabel("Loss")
        train_loss_line, = ax1.plot([], [], label="Training Loss")
        val_loss_line, = ax1.plot([], [], label="Validation Loss")
        ax1.legend()

        ax2.set_title("Accuracy")
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Accuracy")
        acc_line, = ax2.plot([], [], label="Accuracy")
        ax2.legend()

        plt.ion()
        plt.show()

        start_time = time.time()

        for epoch in range(self.max_epochs):  # Use the tunable max_epochs parameter here
            for i in range(num_batches):
                self.conway_model.model.train()
                batch_data = next(train_loader)
                optimizer.zero_grad()
                loss = self.conway_model.model(batch_data)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()

                # Print current batch and epoch
                print(f"Epoch {epoch+1}/{self.max_epochs}, Batch {i+1}/{num_batches}")

                if (i + 1) % self.validate_every == 0:  # Update plots every validate_every batches
                    self.conway_model.model.eval()
                    with torch.no_grad():
                        val_loss = self.conway_model.model(next(cycle(DataLoader(RegressionDataset(self.tokenized_val_data), batch_size=batch_size, drop_last=True)))).item()
                    avg_accuracy = self.evaluate_accuracy()

                    # Update plots
                    train_loss_line.set_xdata(np.append(train_loss_line.get_xdata(), i + epoch * num_batches))
                    train_loss_line.set_ydata(np.append(train_loss_line.get_ydata(), train_loss))
                    val_loss_line.set_xdata(np.append(val_loss_line.get_xdata(), i + epoch * num_batches))
                    val_loss_line.set_ydata(np.append(val_loss_line.get_ydata(), val_loss))
                    acc_line.set_xdata(np.append(acc_line.get_xdata(), i + epoch * num_batches))
                    acc_line.set_ydata(np.append(acc_line.get_ydata(), avg_accuracy))
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # Save metrics
                    metrics.append([epoch, i, train_loss, val_loss, avg_accuracy, time.time() - start_time])

                    # Print loss and accuracy information
                    print(f"Batch {i+1}/{num_batches}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {avg_accuracy}")

                    if avg_accuracy >= 0.999:
                        plt.ioff()
                        plt.show()
                        return avg_accuracy

        plt.ioff()
        plt.show()

        # Save metrics to CSV
        os.makedirs('metrics', exist_ok=True)
        with open(f'metrics/{model_name}_metrics.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Batch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Elapsed Time'])
            writer.writerows(metrics)

        return avg_accuracy

    def optimize(self, space):
        res = gp_minimize(self.objective, space, n_calls=30, random_state=0)
        return res
