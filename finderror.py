
import torch
import numpy as np
def find_error(bias,bias_posit,weights_,weights_posit):
    error = []
    for i in range(3):
        sum = 0
        for j in range(bias[i].shape[1]):
            diff= abs(bias[i].detach().numpy()[0][j]-bias_posit[i].detach().numpy()[0][j])**2
            #if (diff>1):
            
            #print(bias[i].detach().numpy()[0][j])
            #print(bias_posit[i].detach().numpy()[0][j])
            sum = sum + diff
        sum = sum / bias[i].shape[1]
        error.append(sum)
    print('Average Error(Bias): ',(error[0]+error[1]+error[2])/3)
    #print(error)
    error = []
    for i in range(3):
        sum = 0
        for j in range(weights_[i].shape[1]):
            sum = sum + abs(weights_[i].detach().numpy()[0][j]-weights_posit[i].detach().numpy()[0][j])**2
        error.append(sum)


    print('Average Error(Weights): ',(error[0]+error[1]+error[2])/3)
