import torch
from agent.networks import CNN0, CNN1, CNN2

def convert (nparray):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.tensor(nparray).to(device)

class BCAgent:

    def __init__(self, lr, history_length):
        # TODO: Define network, loss function, optimizer
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.net = CNN2(history_length=history_length, n_classes=5).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # Transforming input into tensors
        X_batch =  convert(X_batch)
        y_batch = convert(y_batch)

        # TODO: forward + backward + optimize
        # Forward pass
        y_predictions = self.net(X_batch)
        # Backward pass and Optimizer
        loss = self.loss_function(y_predictions, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        with torch.no_grad():
            outputs = self.net(convert(X))
            return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
