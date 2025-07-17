import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from loader_dataset import * 
import matplotlib.pyplot as plt

# ----- Rete Neurale -----
class NN_torch(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_torch, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ----- Funzione di Addestramento -----
def train_model(model, criterion, optimizer, scheduler, X_train, y_train, num_epochs=10):
    since = time.time()
    train_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())


        print(f'Train Loss: {loss.item():.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')

     # Plot loss
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()

# ----- Funzione di Valutazione -----
def evaluate_model(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
       
        loss = criterion(outputs, y)

    if outputs.shape[1] > 1 and y.ndim == 2:
       y_labels = torch.argmax(y, dim=1)
       preds = torch.argmax(outputs, dim=1)
       accuracy = (preds == y_labels).float().mean().item()
    else:
            accuracy = None

    print(f'Test Loss: {loss.item():.4f}')
    if accuracy is not None:
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

# ----- Main -----
# Carica i dati dal file .npz
data = np.load('test.npz')

# Costruisci tensori
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print("target è",y_test)

# Crea modello
input_size = data['input_size'].item()
output_size = data['output_size'].item()
model = NN_torch(input_size, output_size)

# Inizializza i pesi dal file
with torch.no_grad():
    model.layer1.weight.copy_(torch.tensor(data['layer1_weights'], dtype=torch.float32).T)
    model.layer1.bias.copy_(torch.tensor(data['layer1_bias'], dtype=torch.float32))
    model.layer2.weight.copy_(torch.tensor(data['layer2_weights'], dtype=torch.float32).T)
    model.layer2.bias.copy_(torch.tensor(data['layer2_bias'], dtype=torch.float32))
    model.layer3.weight.copy_(torch.tensor(data['layer3_weights'], dtype=torch.float32).T)
    model.layer3.bias.copy_(torch.tensor(data['layer3_bias'], dtype=torch.float32))

# Imposta ottimizzatore e scheduler
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Allenamento 
train_model(model, criterion, optimizer, scheduler, X_train, y_train, num_epochs=100000)

# Valutazione su Test
evaluate_model(model, criterion, X_test, y_test)
