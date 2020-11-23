import torch
import time
from classification_metrics import binary_eval_metrics, multi_class_eval_metrics


class BaseModel:
    def __init__(self,
                 model,
                 epochs,
                 batch_size,
                 gpu_num=0):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_num}')
            print(f'Training device is gpu:{gpu_num}')
        else:
            self.device = torch.device('cpu')
            print('Training device is cpu')

        self.train_loader, self.valid_loader = self.load_data()
        self.optimizer = self.build_optimizer()

    def run(self, model_path):

        min_valid_loss = float('inf')

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_matrix = self._train_func()
            valid_loss, valid_matrix = self._test_func()

            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model, model_path)
                print(f'\nSave model done valid loss: {valid_loss:.4f}')

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tMetrics: {train_matrix}%(train)')
            # print(f'\tLoss: {train_loss:.4f}(train)\t|\tMetrics: {train_matrix * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tMetrics: {valid_matrix}%(valid)')

    def _train_func(self):
        train_loss = 0
        train_correct = 0
        for step, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device).long(), y.to(self.device)
            output = self.model(x)
            logits = self.calculate_logits(output)
            loss = self.loss_function(logits, y)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_correct += (output.argmax(1) == y).sum().item()

        # if self.scheduler is not None:
        #     self.scheduler.step()

        return train_loss / len(self.train_loader), train_correct / len(self.train_loader.dataset)

    def _test_func(self):
        valid_loss = 0
        valid_correct = 0
        for x, y in self.valid_loader:
            x, y = x.to(self.device).long(), y.to(self.device)
            with torch.no_grad():
                output = self.model(x)
                loss = self.loss_function(output, y)
                valid_loss += loss.item()
                valid_correct += (output.argmax(1) == y).sum().item()

        return valid_loss / len(self.valid_loader), valid_correct / len(self.valid_loader.dataset)

    def load_data(self):
        raise NotImplemented('must be implemented in subclass')

    def loss_function(self, prediction, label):
        raise NotImplemented('must be implemented in subclass')

    def build_optimizer(self):
        raise NotImplemented('must be implemented in subclass')

    def calculate_logits(self, output):
        raise NotImplemented('must be implemented in subclass')

    def train_matrix(self):
        raise NotImplemented('must be implemented in subclass')
