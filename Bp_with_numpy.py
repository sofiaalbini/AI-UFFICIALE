import numpy as np
from loader_dataset import *
import matplotlib.pyplot as plt
class NN_Numpy:
    def __init__(self, init_json):
        self.input_size = init_json['input_size']
        self.output_size = init_json['output_size']
        self.layer1_weights = init_json['layer1_weights']
        self.layer1_bias = init_json['layer1_bias']
        self.layer2_weights = init_json['layer2_weights']
        self.layer2_bias = init_json['layer2_bias']
        self.layer3_weights = init_json['layer3_weights']
        self.layer3_bias = init_json['layer3_bias']

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.layer1_weights) + self.layer1_bias
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.layer2_weights) + self.layer2_bias
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.layer3_weights) + self.layer3_bias
        return self.z3  # Output lineare

    def backward(self, x, y, output, lr=0.001):
        # Derivata MSE: dL/dy_pred = 2*(y_pred - y)
        dloss = 2 * (output - y) / y.shape[0]

        # Layer 3
        dW3 = np.dot(self.a2.T, dloss)
        db3 = np.sum(dloss, axis=0)

        da2 = np.dot(dloss, self.layer3_weights.T)
        dz2 = da2 * self.relu_derivative(self.z2)

        # Layer 2
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.layer2_weights.T)
        dz1 = da1 * self.relu_derivative(self.z1)

        # Layer 1
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update pesi
        self.layer3_weights -= lr * dW3
        self.layer3_bias -= lr * db3
        self.layer2_weights -= lr * dW2
        self.layer2_bias -= lr * db2
        self.layer1_weights -= lr * dW1
        self.layer1_bias -= lr * db1

    def train(self, X, Y, epochs=1000, lr=0.001):
        losses = []
        for epoch in range(epochs):
            
            output = self.forward(X)
           
            loss = np.mean((Y - output) ** 2)
            self.backward(X, Y, output, lr)
            losses.append(loss)
          

          
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses
    
    #def evaluate(self):
            #predictions = np.argmax(self.forward(X_iris), axis=1)
            #correct = (predictions == iris.target)
            #
            # accuracy=np.mean(correct)
def test_model(model, X_test, y_test):
    output = model.forward(X_test)  # shape: (batch, num_classes)

    # Previsione = indice del valore massimo
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y_test, axis=1)  # one-hot → classi

    accuracy = np.mean(predictions == labels) 
    #loss = model.compute_loss(output, y_test)
    loss = np.mean((y_test - output) ** 2)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return loss, accuracy

def plot_losses(losses):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

init_data = np.load("test.npz")
instance = NN_Numpy(init_data)
losses = instance.train(X_train, y_train, epochs=1000)
test_model(instance, X_test, y_test)


# Plot
plot_losses(losses) 