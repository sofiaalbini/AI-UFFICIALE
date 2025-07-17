import numpy as np
from tqdm import trange
from loader_dataset import * 
from  classNN import * 
import matplotlib.pyplot as plt


def plot_losses(losses):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()




def test_model(model, X_test, y_test):
    output = model.forward(X_test)

    # Previsione: classe predetta = indice del valore massimo
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y_test, axis=1)  # y_test è in one-hot

    # Calcolo accuratezza
    accuracy = np.mean(predictions == labels) * 100

    # Calcolo MSE loss
    loss = np.mean((y_test - output) ** 2)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return loss, accuracy 


filename = "test.npz" 


n_epochs = 100
instance = NN_Astar(4,3)
instance.save_weights(filename := 'test.npz')
instance = NN_Astar.init_from_disk(filename)

print('Rete pre neighbors')
iris_forward=instance.forward(X_iris)
predictions = np.argmax(iris_forward, axis=1)
print(predictions)

print('Rete post neighbors')
losses=[]

for epoch in trange(n_epochs):
    
    loss=instance.generate_and_evaluate_neighbours(X_train, y_train)
    losses.append(loss)
    print(f"Epoch {epoch}, Loss: {loss:.4f}") 
    loss,accuracy=test_model(instance,X_test,y_test)
    


plot_losses(losses) 




    




