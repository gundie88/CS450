import pandas as pd
import numpy as np
import math


class Node:
    
    def __init__(self, v = -1, l = "Unknown"):
        self.value = v
        self.label = l
        self.children = []
        self.num_children = 0
    def add_child(self, child_node):
        self.children.append(child_node)
        self.num_children += 1
        return child_node
    
    def has_children(self):
        return (self.get_num_children() > 0)
    
    def get_num_children(self):
        return self.num_children
    
    def __repr__(self, level=0):
        ret = "\t" + "-" * level + repr(self.value)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret
    
    
    
class Decision_Tree:
    head = Node()
    
    def __init__(self, h = -1):
        pass
    
    # Start building the tree
    def build_tree(self, data, tree = -1):
        if(tree == -1):
            tree = self.head
            
        ent = []                # List of entropies for each column
        num_col = data.shape[1] # Get number of columns
        
        if(num_col == 1):
            tree.label = data[1,0]
            return tree
        
        # Get entropies for each column
        for i in range(0, num_col - 1): # All columns but the last (target)
            e = self.get_entropy_of_column(data[1:,i])
            ent.append(e)
            
        # Get highest information gain (or lowest entropy in this case)
        highest_info = min(ent) # Lowest entropy = highest information gain
        index_of_info = ent.index(highest_info) # Get index of the higher information gain column
        tree.label = data[0, index_of_info]
        
        possible_values = np.unique(data[1:,index_of_info])
        
        for val in possible_values:
            #print("Level: {}, value: {}".format(level, val))
            child = Node(val)
#            tree.add_child(Node(val)) # Add each branching point
            new_data = data[np.where(data[:,index_of_info] == val)] # Filter new columns to get entropy for
            new_data = np.insert(new_data, 0, data[0], axis=0)      # Add labels back in (since they got deleted on the line above)
            new_data = np.delete(new_data, index_of_info, axis = 1) # Split column
            
            child = self.build_tree(new_data, tree = child) # Build tree with remaining columns 
            tree.add_child(child)
            
        return tree
    
    
    def get_entropy_of_column(self, column):
        length = len(column)
        
        val, counts = np.unique(column, return_counts=True)
        entropies = []
        weighted_average = 0
        
        for i in range(0, len(val)):
            prob = counts[i] / length
            entropy = -prob * math.log2(prob)
            entropies.append(entropy)
            weighted_average += prob*entropy
            
        
        return weighted_average
    
    
    def predict(self, values, labels):
        current_node = self.head
        prediction = self.head.label
        
        while(current_node.has_children() == True):
            # Find the column that we're checking the value on 
                # So we can determine where to split
            index_of_column = labels.index(current_node.label)
            
            # Get the value of that column
            value = values[index_of_column]
            index_of_child = -1
            
            # Find the branch that has the given value
            for i in range(current_node.get_num_children()):
                if(current_node.children[i].value == value):
                    index_of_child = i
                    break   
            
            # Set new split point
            current_node = current_node.children[index_of_child]
            
            # Set current prediction each time in case there's no more children
            prediction = current_node.label
            
        return prediction
        
        #print(index_to_column)
            
        
def get_data(col_names = []):
    
    data = pd.read_csv("car.data", header=None).as_matrix()
    num_columns = data.shape[1]
    
    if(len(col_names) < num_columns):
        col_names = []
        for i in range(data.shape[1]):
            col_names.append(i)
            
    return data
# Driver function
def main():
    # Get the data
    #col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
    data = get_data()
    # Build the tree
    tree = Decision_Tree()
    tree.build_tree(data)
    
    
    # Predict all rows and test if they are correct
    num_true = 0
    num_false = 0
    for row in data[:,:-1]:
        results = tree.predict(row.tolist(), data[0].tolist())
        predicted_row = row.tolist()
        predicted_row.append(results)
        is_in = np.array(predicted_row) in data 
        
        if(is_in):
            num_true += 1
        else:
            num_false += 1
        #print("Our prediction was: {}!".format(is_in))
             
    print("Accuracy for the whole dataset: {:.2f}%".format(num_true/(num_true+num_false)*100))
    print(tree.head)
    
   
if __name__ == "__main__":
    main()
