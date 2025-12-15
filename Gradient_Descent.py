import numpy as np
import matplotlib.pyplot as plt

def forward(x,w):
    return w*x

def cost(xs,ys,w):
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x,w)
        cost+=(y_pred-y)**2
    return cost/len(xs)

def gradient(xs,ys,w):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)

def main():
    x_data=[1.0,2.0,3.0]
    y_data=[2.0,4.0,6.0]

    w=1.0

    print("Predict (befor training)",4,forward(4,w))

    for epoch in range(100):
        cost_val=cost(x_data,y_data,w)
        grad_val=gradient(x_data,y_data,w)
        w-=0.01*grad_val
        print("Epoch:",epoch,"w=",w,"loss=",cost_val)
    print("Predict (after training)",4,forward(4,w))
    

if __name__=="__main__":
    main()