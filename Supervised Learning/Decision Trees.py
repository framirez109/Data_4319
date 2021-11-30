class Node:
    
    def __init__(self, data_frame, labels=[0,1]):
        self.data_frame = data_frame
        self.labels = labels
        self.impurity = gini(self.data_frame, self.labels)
        
        if self.impurity == 0.0:
            self.pure = True
        else:
            self.pure = False
        
        self.condition = None
        
        self.left_child = None
        self.right_child = None
        
        self.size = len(self.data_frame)
        
        
        root = Node(df)
        left, right, cond = best_split(node)
        root.right_child = right
        root.left_child = left
        root.condition = cond
        
        
        
        
        
        
    #gets the measurment of impurity    
    def gini(data_frame, labels=[0,1]):
        df_list = [list(data_frame["labels"])]
        probs = [df_list.count(y)/len(data_frame) for y in labels]
        return 1 - sum(p**2 for p in probs)
        
    def best_split(node):
        df = node.df
        left.node = None
        right.node = None
        dim = None
        bound = None
        info_gain = 0.0
        X = list(df["x0"]).sort()
        Y = list(df["x1"]).sort()
        for x, y in zip(X,Y):
            left = Node(df[df["x0"] <= x])
            right = Node(df[df["x0"] > x])
            ig = node.impurity - ((self.size/node.size)*left.impurity + (right.size/node.size)*right.impurity)
            if ig > info_gain:
                info_gain = ig
                dim = 0
                left.node = left
                right.node = right
                
        #still have to do it for y value
        
        return left.node, right_node (dim,bound)
    
    
######################################

#you have your condition dim, bound (dimension, bound)
def predict(self, x):
    dim, bound = self.condition
    if x[dim] <= bound:
        return self.left_child
    else:
        return self.right_child
    
                
            
            
