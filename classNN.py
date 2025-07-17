from __future__ import annotations
import numpy as np 
from  datetime import datetime


class NN_Astar:
    MIN_VALUE_WEIGHTS = -1
    MAX_VALUE_WEIGHTS = 2  # +1 perché randint esclude l'ultimo valore

    def __init__(self, input_size : int , output_size : int, init_json = None):

        if not init_json:            
            self.input_size=input_size
            self.output_size=output_size
            self.layer1_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (input_size, 64)) * .01
            self.layer1_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (64,)) * .01
            self.layer2_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (64,64)) * .01
            self.layer2_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (64,)) * .01
            self.layer3_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (64, output_size)) * .01
            self.layer3_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (output_size,)) * .01
            return
        self.input_size = init_json['input_size']
        self.output_size = init_json['output_size']
        self.layer1_weights = init_json['layer1_weights']
        self.layer1_bias = init_json['layer1_bias']
        self.layer2_weights = init_json['layer2_weights']
        self.layer2_bias = init_json['layer2_bias']
        self.layer3_weights = init_json['layer3_weights']
        self.layer3_bias = init_json['layer3_bias']

    @staticmethod
    def init_from_disk(filename) -> NN_Astar:
        """Carica i pesi e i bias da un file .npz e li assegna al modello."""
        data = np.load(filename)
        #print("initialize layer_1-weights",data['layer1_weights'])

        print(f"Pesi e bias caricati da: {filename}")
        return NN_Astar(init_json=data, input_size=None, output_size=None)

    def save_weights(self, filename = f"net_{datetime.today()}.npz"):
        """Salva i pesi e i bias su disco in formato .npz."""
        np.savez(
            filename,
            input_size=self.input_size,
            output_size=self.output_size,
            layer1_weights=self.layer1_weights,
            layer1_bias=self.layer1_bias,
            layer2_weights=self.layer2_weights,
            layer2_bias=self.layer2_bias,
            layer3_weights=self.layer3_weights,
            layer3_bias=self.layer3_bias
        )
        print(f"Pesi e bias salvati in {filename}")

    def forward(self, x):
        def relu(x):
            return np.maximum(0, x)  # Funzione di attivazione discreta

        x = relu(np.dot(x, self.layer1_weights) + self.layer1_bias)
        x = relu(np.dot(x, self.layer2_weights) + self.layer2_bias)
        x = np.dot(x, self.layer3_weights) + self.layer3_bias  # Nessuna attivazione in output
        return x
    
    def forward_with_neighbor(self, x, neighors):
        def relu(x):
            return np.maximum(0, x)  # Funzione di attivazione discreta

        x = relu(np.dot(x, neighors[0]) + neighors[1])
        x = relu(np.dot(x, neighors[2]) + neighors[3])
        x = np.dot(x, neighors[4]) + neighors[5]  # Nessuna attivazione in output
        return x

    def generate_and_evaluate_neighbours(self, x, target): 
        n_neighbour=30
        # np.random.seed(seed)
        means_sums_variances = {}
        total_function_costs = []
        param_heuristic, param_original_cost = np.random.randint(-1, 2) * 0.1, 1
        neigbour_layer1_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (n_neighbour,self.input_size, 64))*.01
        #print("noise matrix " , neigbour_layer1_weights)
        neighbour_layer1_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (n_neighbour,64,)) *.01
        neighbour_layer2_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (n_neighbour,64,64)) *.01
        neighbour_layer2_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (64,))*.01
        neighbour_layer3_weights = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (n_neighbour,64, self.output_size))*.01
        neighbour_layer3_bias = np.random.randint(self.MIN_VALUE_WEIGHTS, self.MAX_VALUE_WEIGHTS, (n_neighbour,self.output_size,))*.01

        new_neighbours_1_weights = self.layer1_weights + neigbour_layer1_weights 
        new_neighbours_1_bias = self.layer1_bias + neighbour_layer1_bias 
        new_neighbours_2_weights = self.layer2_weights + neighbour_layer2_weights 
        new_neighbours_2_bias = self.layer2_bias + neighbour_layer2_bias 
        new_neighbours_3_weights = self.layer3_weights + neighbour_layer3_weights 
        new_neighbours_3_bias = self.layer3_bias + neighbour_layer3_bias 

        all_new_neigbors = {
            "new_neighbours_1_weights" : new_neighbours_1_weights,
            "new_neighbours_1_bias" : new_neighbours_1_bias,
            "new_neighbours_2_weights" : new_neighbours_2_weights,
            "new_neighbours_2_bias" : new_neighbours_2_bias,
            "new_neighbours_3_weights" : new_neighbours_3_weights,
            "new_neighbours_3_bias" : new_neighbours_3_bias,
        }

        for idx in range(n_neighbour):

            means_sums_variances[idx] = sum([float(np.sum(neighbor_vector[idx])) for neighbor_vector in all_new_neigbors.values()])

            # means_sums_variances[idx] = {
            #     "sum" : {
            #         neighbor_label : np.sum(neighbor_vector[idx]) 
            #         for neighbor_label, neighbor_vector in all_new_neigbors.items()
            #     }, 
            #     "mean" : {
            #         neighbor_label : np.mean(neighbor_vector[idx]) 
            #         for neighbor_label, neighbor_vector in all_new_neigbors.items()
            #     }, 
            #     "var" : {
            #         neighbor_label : np.var(neighbor_vector[idx]) 
            #         for neighbor_label, neighbor_vector in all_new_neigbors.items()
            #     }, 
            # }



        
            def original_cost_function(data):
                
                return data if data > 0 else - data # Valore assoluto della variazione dei pesi
            
            def heuristic(output, target):
                ## MSE
                return np.mean((target - output) ** 2) 
            
            def heuristic1(output,target):
                epsilon = 1e-10
                output= np.clip(output, epsilon, 1. - epsilon)
                return -np.sum(target * np.log(output)) /target.shape[0] 


            
            # dobbiamo ora applicare il forward che accetta i pesi della rete a cui si è sommato il vicino
            output_per_neighbor = self.forward_with_neighbor(x, [neighbor_vector[idx] for neighbor_vector in all_new_neigbors.values()])
            heuristic_function_cost = heuristic(np.argmax(output_per_neighbor), target) 
            target_function_cost = original_cost_function(means_sums_variances[idx])
            #scaled_heuristic_function_cost =  heuristic_function_cost
            #scaled_target_function_cost = param_orig target_function_cost
            # print(f'idx_neighbor = {idx}, {heuristic_function_cost = }')
            # print(f'idx_neighbor = {idx}, {target_function_cost = }')
            total_function_costs.append(heuristic_function_cost+ target_function_cost)
            #total_function_costs.append(heuristic_function_cost)
        
        # print(total_function_costs)

       # best_neighbor_idx = np.argmin(total_function_costs)
        best_neighbor_idx = np.argmin(total_function_costs)
        best_loss = float(total_function_costs[best_neighbor_idx])
        

        self.layer1_weights = new_neighbours_1_weights[best_neighbor_idx]
        self.layer1_bias = new_neighbours_1_bias[best_neighbor_idx]
        self.layer2_weights = new_neighbours_2_weights[best_neighbor_idx]
        self.layer2_bias = new_neighbours_2_bias[best_neighbor_idx]
        self.layer3_weights = new_neighbours_3_weights[best_neighbor_idx]
        self.layer3_bias = new_neighbours_3_bias[best_neighbor_idx] 

        return best_loss

