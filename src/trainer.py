import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

class Trainer:

    def __init__(self, model, dataset):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        # split train / val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=64)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def evaluate(self):

        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for X, y in self.val_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)

                loss = self.criterion(pred, y)
                total_loss += loss.item()

                _, predicted = torch.max(pred, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = correct / total
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, acc

    def train(self, epochs=30):

        best_acc = 0

        for epoch in range(epochs):

            self.model.train()

            total_loss = 0

            for X, y in self.train_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)

                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(self.train_loader)

            val_loss, val_acc = self.evaluate()

            print(f"Epoch {epoch+1}")
            print(f"Train Loss: {round(train_loss,4)}")
            print(f"Val Loss: {round(val_loss,4)} | Val Acc: {round(val_acc,4)}")
            print("-----")

            # save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "models/best_model.pth")