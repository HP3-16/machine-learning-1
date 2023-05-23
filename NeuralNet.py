import numpy as np

class NeuralNetwork:
    def __init__(self,layers, eta =0.01):
        """
        The layer architecture along with the learning rate is initiated
        """
        self.W = []
        self.layers = layers
        self.eta = eta

        for i in np.arange(0,len(layers)-2):
            wt = np.random.randn(layers[i]+1,layers[i+1]+1)
            norm_wt = wt/np.sqrt(layers[i])
            self.W.append(norm_wt)
        
        wt = np.random.randn(layers[-2]+1,layers[-1])
        norm_wt = wt/np.sqrt(layers[-2])
        self.W.append(norm_wt)
    
    def sigmoid(self,x):
        """
        returns the sigmoid activation value
        """
        return 1.0/(1+np.exp(-x))
    
    def sigmoid_grad(self,x):
        return x*(1-x)
    
    def fit(self,X,y,epochs=100,display=100):
        """
        Update the weights
        """
        X = np.c_(X,np.ones((X.shape[0]))) # add bias to train 

        for epoch in np.arange(0,epochs):
          for(x,target) in zip(x,y):
              self.fit_partial(x,target)
        
        if epoch == 0 or (epoch+1)% display == 0:
            loss = self.calc_loss(X,y)
            print("[INFO] epoch={}, loss={:.7f}".format(epoch+1,loss))



