import csv
import numpy as np
import random
import math

INPUT_LAYER = 2    #NO. of nodes in input layer
HIDDEN_LAYER = 2   #NO. of nodes in hidden layer
OUTPUT_LAYER = 1   #NO. of nodes in otput layer
num_labels = 1
learning_rate = 0.1
def sigmoidGradient(z):
    t = 1 - sigmoid(z)
    sg = np.multiply(sigmoid(z),t)                 #Elementwise multiplication of matrices
    return sg

def sigmoid(z):
    g = 1.0  / (1.0 + np.exp(-1.0 * z))
    return g

def costfunction(X,Y,h,Theta_1,Theta_2):
   m = np.shape(X)
   i = [1] - h
   j = []
   for a in range(len(Y)):
       j.append(1.0 - Y[a])
   J = 0
   J = (1/4)*(-1.0*np.matmul(np.log(h),Y) - np.matmul(np.log(i),j))   #Without regularisation
   return J

def Randomise():
    EPSILON_INIT = 0.12
    Theta_1 = np.random.rand(HIDDEN_LAYER,INPUT_LAYER+1)*(2*EPSILON_INIT)-(EPSILON_INIT)     # Random Weights for level 1
    Theta_2 = np.random.rand(OUTPUT_LAYER,HIDDEN_LAYER+1)*(2*EPSILON_INIT)-(EPSILON_INIT)    # Random Weights for level 2
    return Theta_1,Theta_2

def Neural(X,Theta_1,Theta_2,Y):
    """forward propogation"""
    Theta1_grad = np.zeros(np.shape(Theta_1))
    Theta2_grad = np.zeros(np.shape(Theta_2))
    m = np.shape(X)
    a1 = []
    X = np.insert(X,0,1,axis=1)
    z2 = np.matmul(Theta_1,np.transpose(X))
    a2 = sigmoid(z2)
    a2 = np.transpose(a2)
    a2 = np.insert(a2,0,1,axis=1)
    z3 = np.matmul(Theta_2,np.transpose(a2))
    h = sigmoid(z3)
    J = costfunction(X,Y,h,Theta_1,Theta_2)
    temp = 0.0
    flag = 0
    while(J - temp >0.0001):
        for a in range(4):
            temp = J
            a1 = X[a]                                        #Forward Propogation
            z2 = np.matmul(Theta_1,np.transpose(a1))
            a2 = np.transpose(sigmoid(z2))
            a2 = np.insert(a2,0,1)
            z3 = np.matmul(Theta_2,a2)
            a3 = sigmoid(z3)
            z2 = np.transpose(z2)
            z2 = np.insert(z2,0,1)
            delta_3 = a3 - Y[a]                              #Backward Propogation
            p = np.matmul(np.transpose(Theta_2),delta_3)
            q = sigmoidGradient(z2)
            delta_2 = np.multiply(p,q)                       #Elementwise multiplication of matrices
            delta_2 = [delta_2[1],delta_2[2]]
            delta_3 = [[delta_3[0]]]
            a2 = [[a2[0]], [a2[1]], [a2[2]]]                 #Converting list to matrix
            Theta2_grad = Theta2_grad + np.sum(np.matmul(delta_3,np.transpose(a2)))
            a1 = [[a1[0]],[a1[1]],[a1[2]]]
            delta_2 = [[delta_2[0]],[delta_2[1]]]
            Theta1_grad = Theta1_grad + np.matmul(delta_2,np.transpose(a1))
            Theta_1 = Theta_1 - (learning_rate/4) * np.matmul(delta_2,np.transpose(a1))
            Theta_2 = Theta_2 - (learning_rate/4) * np.matmul(delta_3,np.transpose(a2))
            J = costfunction(X,Y,h,Theta_1,Theta_2)

    return temp,J,Theta_1,Theta_2

def main():
    X = []
    Y = []
    with open("XOR.csv","r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)):
            X.append([int(dataset[i][0]),int(dataset[i][1])])
            Y.append(int(dataset[i][2]))
    Theta_1,Theta_2 = Randomise()
    temp,J,Theta_1,Theta_2 = Neural(X,Theta_1,Theta_2,Y)
    print(J)
    print(temp)
    data = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    pred = sigmoid(np.matmul(data,np.transpose(Theta_2)))
    print(pred)
main()
