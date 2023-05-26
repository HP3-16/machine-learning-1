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
        
        #last layer doesn't need a bias but the penultimate one requires bias
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
              self.fit_partial(x,target) #Back prop
        
        if epoch == 0 or (epoch+1)% display == 0:
            loss = self.calc_loss(X,y)
            print("[INFO] epoch={}, loss={:.7f}".format(epoch+1,loss))

    def fit_partial(self,x,y):
        A = np.atleast_2d(x) #Activation list

        #FeedForward Network
        for layer in np.arange(0,len(self.W)):
            pre_out = A[layer].dot(self.W[layer])
            out = self.sigmoid(pre_out)
            A.append(out)

        #BACKPROP
        error = A[-1]-y
        D = [] #Derivative List
        D.append(error * self.sigmoid_grad(A[-1]))

        for layer in np.arange(len(A)-2, 0, -1):
            grad = D[-1].dot(self.W[layer].T)
            grad = grad * self.sigmoid_grad(A[layer])
            D.append(grad)


        D = D[::-1]

        #Weight updation:
        for layer in np.arange(0,len(self.W)):
            self.W[layer] = self.W[layer] - self.eta * A[layer].T.dot(D[layer])

    def predict(self,X,Bias=True):
        p = np.atleast_2d(X)

        if Bias:
            p = np.c_(p,np.oness((p.shape[0])))

        for layer in np.arange(0,len(self.W)):
            p = self.sigmoid(np.dot(p,self.W[layer]))
        
        return p

    def calc_loss(self,X,targets):
        targets = np.atleast_2d(targets)
        pred = self.predict(X,Bias=False)
        loss = 0.5 * np.sum((pred-targets)**2) #MSE
        return loss



