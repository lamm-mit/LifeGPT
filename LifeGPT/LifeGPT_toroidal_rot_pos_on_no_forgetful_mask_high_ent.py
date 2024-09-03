print('Training LifeGPT')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import seaborn as sns
import time
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import datetime
from IPython.display import display, HTML
from tqdm import tqdm, trange
import typing
import matplotlib.pyplot as plt
import csv

# Function to clear CUDA cache
def empty_cuda_cache():
    torch.cuda.empty_cache()

# Ensure Torch is using Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Torch version:", torch.__version__)
empty_cuda_cache()

data_path = "LifeGPT\\"
train_file = data_path + "conway_states_0.5_0.5_10000by32by32by10_toroidal_20240813_224815_sorder0.5_eorder0.5"+".csv"
val_file = data_path + "conway_states_0.5_0.5_1000by32by32by10_toroidal_20240813_224611_sorder0.5_eorder0.5"+".csv"
test_file = data_path + "conway_test_states_32by32_20240716_151502.csv"


# Load in Data from CSVs
df_train = pd.read_csv(train_file)
df_val = pd.read_csv(val_file)
df_test = pd.read_csv(test_file)

# Define training size and special characters
TRAIN_SIZE = 10000
TEST_SIZE = 1000
start_char = '@'
end_char = '$'
mask_char = ['_']

# Function to generate data
def generate_data(df, future_steps):
    X_data = []
    for i in range(len(df['State 1'])):
        future_state_col = f'State {future_steps}'
        if future_state_col in df.columns:
            str_ = f"{start_char}PredictNextState<{df['State 1'][i]}> [{df[future_state_col][i]}]{end_char}"
            X_data.append(str_)
    return X_data

# Generate datasets for different future steps
future_steps_list = [2, 3, 5, 10]
X_data_train = {steps: generate_data(df_train, steps) for steps in future_steps_list}
X_data_val = {steps: generate_data(df_val, steps) for steps in future_steps_list}
X_data_test = {steps: generate_data(df_test, steps) for steps in future_steps_list}
print(X_data_test[2][0])
print(X_data_test[2][0][0])

# Print the number of sequences in each dataset
for steps in future_steps_list:
    print(f"Train set for {steps} future steps: {len(X_data_train[steps])} sequences")
    print(f"Validation set for {steps} future steps: {len(X_data_val[steps])} sequences")

# Find the maximum sequence length across all training datasets
max_length = max(len(seq) for steps in future_steps_list for seq in X_data_train[steps])
print('Max length sequence:', max_length)

# Initialize Tokenizer
num_words = 256

class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad = n_pad
        self.device = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad, f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(self, texts: typing.List[str], encoding="utf8", do_padding=True):
        sentences = [self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0) for sentence in texts]
        return torch.cat(sentences, dim=0).to(self.device)

    @staticmethod
    def prepare_texts(document: str) -> typing.List[str]:
        return filter(lambda x: len(x) != 0, document.split("\n"))

    def sequences_to_texts(self, texts: torch.Tensor, encoding="utf8"):
        out = []
        for seq in texts:
            chars = []
            i = 0
            while i < len(seq) and seq[i] != 0:
                chars.append(int(seq[i]))
                i += 1
            try:
                out.append(bytes(chars).decode(encoding))
            except:
                pass
        return out

# Initialize tokenizer
tokenizer_X = Tokenizer(max_length, device)

# Function to tokenize data and print an example
def tokenize_data(data, tokenizer):
    tokenized_data = tokenizer.texts_to_sequences(data)
    print('Example tokenized data:', tokenized_data[0])
    return tokenized_data

# Tokenize training and validation data for all future steps
X_data_train_tokenized = {steps: tokenize_data(X_data_train[steps], tokenizer_X) for steps in future_steps_list}
X_data_val_tokenized = {steps: tokenize_data(X_data_val[steps], tokenizer_X) for steps in future_steps_list}

# Function to tokenize and print special character tokens
def tokenize_special_char(char, tokenizer):
    token = tokenizer.texts_to_sequences([char])
    token_value = token[0][0].cpu().numpy()
    print(f'{char} token:', token_value)
    return token_value

# Define special character tokens
start_char_token = tokenize_special_char(start_char, tokenizer_X)
end_char_token = tokenize_special_char(end_char, tokenizer_X)
mask_token = tokenize_special_char(mask_char[0], tokenizer_X)

print('Mask token:', mask_token)

def remove_start_end_token(string_input, start='@', end='$'):
    res = string_input.replace(start, "").replace(end, "")
    return res

def remove_start_end_token_first(string_input, start='@', end='$'):
    i = string_input.find(start)
    j = string_input.find(end)
    return string_input[i+1:j]

def extract_task(string_input, end_task_token=')'):
    j = string_input.find(end_task_token)
    return string_input[:j+1]

def extract_start_and_end(string_input, start_token='[', end_token=']'):
    i = string_input.find(start_token)
    j = string_input.find(end_token)
    return string_input[i+1:j]

# Initialize Reverse Tokenizer
def reverse_tokenize(tokenizer_X, X_data, X_norm_factor=1):
    X_data_tokenized_reversed = tokenizer_X.sequences_to_texts((X_data * X_norm_factor).int())
    return [i for i in X_data_tokenized_reversed]

aa = reverse_tokenize(tokenizer_X, X_data_train_tokenized[2][1:2])
bb = X_data_train[2][1:2]
if aa == bb:
    print('Tokenization and reverse tokenization consistent')
else:
    print('Inconsistent behavior')

print(extract_task(remove_start_end_token_first(X_data_train[2][1]), end_task_token='>'))
print(X_data_train[2][1])
print(extract_start_and_end(X_data_train[2][1], start_token='<', end_token='>'))
print(extract_start_and_end(X_data_train[2][1], start_token='[', end_token=']'))
print(reverse_tokenize(tokenizer_X, X_data_val_tokenized[2][:1]))
print(X_data_val[2][1])

# Initialize the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Function to clear CUDA cache
def empty_cuda_cache():
    torch.cuda.empty_cache()

# Updated get_model function with model naming
def get_model(max_length, num_words, model_name, dim=256, depth=12, heads=8, attn_dim_head=64, rot_pos=True, attn_flash=True, masking=True, mask_prob = 0.15):
    empty_cuda_cache()

    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dim_head=attn_dim_head,
            rotary_pos_emb=rot_pos,
            attn_flash=attn_flash
        )
    )

    if masking == True:
        model = AutoregressiveWrapper(model, mask_prob = mask_prob)
        model.cuda()
        print(f'model created with rot pos enc as {rot_pos}, attn flash as {attn_flash}, and masking set to {masking}')
    else: 
        model = AutoregressiveWrapper(model)
        model.cuda()
        print(f'model created with rot pos enc as {rot_pos}, attn flash as {attn_flash}, and masking set to {masking}')

    # Get the current time
    model_creation_time = datetime.datetime.now()
    model_creation_time_str = model_creation_time.strftime("%Y-%m-%d %H-%M-%S")

    # Print model creation time
    print(f'Model "{model_name}" Created @ {model_creation_time_str} Eastern Time')

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")

    # Create directory for model parameters
    model_dir = os.path.join("model_parameters", f"{model_name}_{model_creation_time_str}")
    os.makedirs(model_dir, exist_ok=True)

    # Write model details to a .txt file
    model_info_file = os.path.join(model_dir, f"{model_name}_info.txt")
    with open(model_info_file, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Created @ {model_creation_time_str} Eastern Time\n")
        f.write(f"Number of trainable parameters: {num_params}\n")
        f.write(f"Model Architecture:\n{model}\n")
        f.write("Model Parameters:\n")
        f.write(f"num_tokens: {num_words}\n")
        f.write(f"max_seq_len: {max_length}\n")
        f.write(f"dim: {dim}\n")
        f.write(f"depth: {depth}\n")
        f.write(f"heads: {heads}\n")
        f.write(f"attn_dim_head: {attn_dim_head}\n")
        f.write(f"rotary_pos_emb: {rot_pos}\n")
        f.write(f"attn_flash: {attn_flash}\n\n")
        f.write("Note:Aug 14, 2024 - this model is a test using rot pos enc, no extra masking, and high entropy data on a 32 by 32 toroidal grid. Poor performance on the test set is anticipated.")

    return model, model_dir

model_name = "08_14_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_Off_High_Entropy_Homog"
rot_pos = True
model, model_dir = get_model(max_length, num_words, model_name, rot_pos=rot_pos, masking=False, mask_prob = 0) ##EDIT THIS FOR MODEL PARAMS

empty_cuda_cache()
print("Cuda cache has been cleared")

LEARNING_RATE = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
# Function to cycle through data loader indefinitely
def cycle(loader):
    while True:
        for data in loader:
            yield data

# RegressionDataset class definition
class RegressionDataset(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    
    def __getitem__(self, index):
        return self.X_data[index]
    
    def __len__(self):
        return len(self.X_data)

# NUM_BATCHES = 10 <--- not needed in this version of the code
NUM_EPOCHS = 50
SAVE_EPOCH = 2
VALIDATE_EVERY = 5
GENERATE_EVERY = 10
GRADIENT_ACCUMULATE_EVERY = 5
# GENERATE_LENGTH = 904
GENERATE_LENGTH = max_length - len(extract_task(X_data_train[steps][0],end_task_token='>'))
print("generate_length = ", GENERATE_LENGTH)
BATCH_SIZE = 20
MEASURE_ACC_EVERY = 1000

steps = future_steps_list[0]  # Initialize with the first element of future_steps_list
print(f"Predicting jump from state 1 to state {steps}")
print(f"Predicting jump from state 1 to state {steps}")
print(f"Predicting jump from state 1 to state {steps}")
print(f"Predicting jump from state 1 to state {steps}")
print(f"Predicting jump from state 1 to state {steps}")

# Create train and validation datasets and loaders
train_dataset = RegressionDataset(X_data_train_tokenized[steps])
val_dataset = RegressionDataset(X_data_val_tokenized[steps])
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))
train_sample = next(train_loader)
val_sample = next(val_loader)
print(f'Train batch shape: {train_sample.shape}')
print(f'Validation batch shape: {val_sample.shape}')

torch.cuda.empty_cache()

# Disable the output's buffering in Jupyter notebook
display(HTML("<style>.output_result { max-height:10000px; }</style>"))

# Function to generate samples
def generate_sample():
    model.eval()
    inp = torch.Tensor(tokenizer_X.texts_to_sequences(extract_task(random.choice(X_data_val), end_task_token='>'), do_padding=False)).to(device)
    inp = inp.transpose(0, 1)
    inp = inp.long()
    prime = (reverse_tokenize(tokenizer_X, inp[:1])[0])

    sample = model.generate(
        prompts=inp,
        seq_len=GENERATE_LENGTH,
        cache_kv=True
    )
    try:
        output_str = reverse_tokenize(tokenizer_X, sample[:1])
    except:
        output_str = "non utf token found in sample output"
    return output_str

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import time  # Import time module

def count_mismatches(ground_truth, pred):
    mismatches = sum(1 for gt, p in zip(ground_truth, pred) if gt != p)
    accuracy = 1 - mismatches / len(ground_truth)
    return mismatches, accuracy

# Initialize the CSV file in the model parameters folder
csv_file_path = os.path.join(model_dir, 'loss_data.csv')
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "batch_within_epoch", "batch_overall", "train_loss", "val_loss", "accuracy", "elapsed_time"])

# Training loop
train_losses = []
val_losses = []
batches = []
accuracies = []
accuracies_batches = []
epoch_list = []
batch_overall = []

# Set up the interactive plots
plt.ion()
fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 10))

# Training and Validation Losses plot
train_loss_line, = ax_loss.plot([], [], label="Training Loss")
val_loss_line, = ax_loss.plot([], [], label="Validation Loss")
ax_loss.set_xlabel("Batch Overall")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Training and Validation Losses")
ax_loss.legend()
ax_loss.grid(True)

# Model Accuracy plot
acc_line, = ax_acc.plot([], [], label="Accuracy")
ax_acc.set_xlabel("Batch Overall")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Model Accuracy")
ax_acc.legend()
ax_acc.grid(True)

# Adding secondary x-axis for epochs
ax_loss_epoch = ax_loss.twiny()
ax_acc_epoch = ax_acc.twiny()

# Plot customization for secondary x-axis
plt.show()

total_batches = NUM_EPOCHS * int(TRAIN_SIZE / BATCH_SIZE)
overall_batch_counter = 0
accuracy = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    num_batches = int(TRAIN_SIZE / BATCH_SIZE)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_start_time = time.time()  # Track the start time of the epoch
    for i in range(num_batches):
        batch_start_time = time.time()  # Track the start time of the batch

        model.train()
        loss = model(next(train_loader))
        loss.backward()
        
        if (i + 1) % GRADIENT_ACCUMULATE_EVERY == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = model(next(val_loader)).item()

        # Store the losses and batch/epoch info
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        batches.append(overall_batch_counter)
        batch_overall.append(overall_batch_counter)

        # Calculate accuracy
        if (i+1) % MEASURE_ACC_EVERY == 0:
            valid_output_found = False
            attempt_count = 0
            max_attempts = 1  # Maximum attempts to get a valid output before skipping
            accuracy_list = []
            while len(accuracy_list) < 10 and attempt_count < max_attempts:
                inp_seq = X_data_test[2][len(accuracy_list)]
                inp = extract_task(inp_seq, end_task_token='>')
                inp = torch.Tensor(tokenizer_X.texts_to_sequences(inp, do_padding=False)).to(device)
                inp = inp.transpose(0, 1).long()
                with torch.no_grad():
                    sample = model.generate(
                                prompts=inp,
                                seq_len=GENERATE_LENGTH,
                                cache_kv=True
                            )
                try:
                    output_str = reverse_tokenize(tokenizer_X, sample[:1])
                    pred = extract_start_and_end(output_str[0], start_token='[', end_token=']')
                    ground_truth = extract_start_and_end(inp_seq, start_token='[', end_token=']')
                    _, acc = count_mismatches(ground_truth=ground_truth, pred=pred)
                    accuracy_list.append(acc)
                    valid_output_found = True
                except Exception as e:
                    print(f"Error decoding output: {e}")
                    attempt_count += 1
            
            if valid_output_found and len(accuracy_list) > 0:
                accuracy = np.mean(accuracy_list)
            else:
                accuracy = 0  # In case of failure to decode valid output

            accuracies.append(accuracy)
            accuracies_batches.append(overall_batch_counter)

        # Append loss and accuracy data to the CSV file
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, i, overall_batch_counter, loss.item(), val_loss, accuracy, elapsed_time])

        # Update the plots
        train_loss_line.set_xdata(batch_overall)
        train_loss_line.set_ydata(train_losses)
        val_loss_line.set_xdata(batch_overall)
        val_loss_line.set_ydata(val_losses)
        
        # Update accuracy plot only if accuracies list is updated
        if len(accuracies_batches) > 0:
            acc_line.set_xdata(accuracies_batches)
            acc_line.set_ydata(accuracies)
        
        ax_loss.relim()
        ax_loss.autoscale_view()
        ax_acc.relim()
        ax_acc.autoscale_view()
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

        # Print loss and accuracy information
        if (i + 1) % VALIDATE_EVERY == 0:
            print(f"Batch {i+1}/{num_batches}, Train Loss: {loss.item()}, Validation Loss: {val_loss}, Accuracy: {accuracy}")
        
        overall_batch_counter += 1

    # Save the model periodically
    if (epoch + 1) % SAVE_EPOCH == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'LifeGPT_epoch_{epoch + 1}.pt'))
        print(f'Model saved at epoch {epoch + 1}')

    epoch_end_time = time.time()  # Track the end time of the epoch
    print(f"Epoch {epoch+1} time: {epoch_end_time - epoch_start_time:.2f} seconds")

# Update secondary x-axis for epochs
epochs_ticks = np.arange(0, total_batches, num_batches)
ax_loss_epoch.set_xticks(epochs_ticks)
ax_loss_epoch.set_xticklabels(np.arange(1, NUM_EPOCHS + 1))
ax_loss_epoch.set_xlabel("Epochs")

ax_acc_epoch.set_xticks(epochs_ticks)
ax_acc_epoch.set_xticklabels(np.arange(1, NUM_EPOCHS + 1))
ax_acc_epoch.set_xlabel("Epochs")

# Save the final plots
plt.ioff()
fig.savefig(os.path.join(model_dir, 'loss_accuracy_plot.png'))
plt.show()
