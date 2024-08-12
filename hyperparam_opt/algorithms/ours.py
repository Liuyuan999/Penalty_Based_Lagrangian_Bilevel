import numpy as np
import time

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split

def lag_F(w, b, mu, C, gamma, y_val, z_val, y_train, z_train):
    x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
    x = x* F.linear(torch.Tensor(z_val), w, b)
    loss_upper= torch.sum(torch.exp(1-x))

    loss_lower = (1/2) * (1/2) * ((w**2).sum() +b**2)

    restr_lower = 1 - C - y_train * F.linear(z_train, w, b).squeeze()

    return loss_upper + gamma*loss_lower + mu @ restr_lower

def lag_g(w, b, mu, C, gamma, y_val, z_val, y_train, z_train):

    loss_lower = (1/2) * ((w**2).sum() +b**2)

    restr_lower = 1 - C - y_train * F.linear(z_train, w, b).squeeze()

    return loss_lower + mu @ restr_lower

def minmax_opt(w, b, mu, lag_fn, C, gamma, eta1, eta2, T, Ty, y_val, z_val, y_train, z_train):

    for t in range(T):

        mu.requires_grad_(False)
        w.requires_grad_(True)
        b.requires_grad_(True)

        for ty in range(Ty):
            lag_val = lag_fn(w, b, mu, C, gamma, y_val, z_val, y_train, z_train)

            # Reset grads
            w.grad = None
            b.grad = None

            lag_val.backward()
            grad_w = w.grad.detach().clone()
            grad_b = b.grad.detach().clone()

            #print(f"{lag_val=} {torch.linalg.norm(grad_w)=}")
            w.data -= eta1*grad_w
            b.data -= eta1*grad_b
        

            if torch.linalg.norm(grad_w) + torch.linalg.norm(grad_b) < 1e-4:
                break

        mu.requires_grad_(True)
        w.requires_grad_(False)
        b.requires_grad_(False)
        lag_val = lag_fn(w, b, mu, C, gamma, y_val, z_val, y_train, z_train)

        #print(f"LAG VAL IN OUTER {lag_val=}")
        mu.grad = None
        lag_val.backward()
        grad_mu = mu.grad.detach().clone()
        #print(f"{grad_mu=}")
        mu.data = torch.maximum(torch.tensor(1e-5), mu.data + eta2*grad_mu)
        if torch.linalg.norm(grad_mu) < 1e-4:
            break

    mu.requires_grad_(False)
    w.requires_grad_(False)
    b.requires_grad_(False)

    #print(w)

    return w, b, mu

def ours(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, early_stopping_th = False,verbose=True):
    feature=x_train.shape[1] # = 8

    # Dataset to tensor
    y_train = torch.tensor(y_train).float()
    x_train = torch.tensor(x_train).float()
    y_val = torch.tensor(y_val).float()
    x_val = torch.tensor(x_val).float()
    y_test = torch.tensor(y_test).float()
    x_test = torch.tensor(x_test).float()
    
    ###### Parameters
    eta = hparams['eta']
    eta1g = hparams['eta1g']
    eta2g = hparams['eta2g']
    eta1F = hparams['eta1F']
    eta2F = hparams['eta2F']
    gam = hparams['gam']
    T = hparams['T']
    Ty = hparams['Ty']

    # Initialization of upper and lower level variables
    C_tensor_val= torch.Tensor(x_train.shape[0]).uniform_(1.,5.)

    wg = torch.zeros(1,feature)
    bg = torch.tensor(1.)
    xig = torch.zeros(y_train.shape[0])
    mug = torch.zeros(y_train.shape[0])

    wF = wg.clone()
    bF = bg.clone()
    xiF = xig.clone()
    muF = mug.clone()
    
    # For storage
    val_loss_list=[]
    test_loss_list=[]
    val_acc_list=[]
    test_acc_list=[]
    time_computation=[]
    algorithm_start_time=time.time()

    metrics = []
    variables = []

    for epoch in range(epochs):

        variables.append({
            'C': C_tensor_val,
            'xi': xig,
            'w': wg,
            'b': bg,
            'xi_F': xiF,
            'w_F': wF,
            'b_F': bF
        })

        x = torch.reshape(y_val, (y_val.shape[0],1)) 
        x = x* F.linear(x_val, wF, bF) # / torch.linalg.norm(w_tensor)

        x1 = torch.reshape(y_test, (y_test.shape[0],1)) 
        x1 = x1 * F.linear(x_test, wF, bF) # / torch.linalg.norm(w_tensor)
        # test_loss_upper= torch.sum(torch.sigmoid(x1))
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss_F = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss_F = test_loss_upper.detach().numpy()/y_test.shape[0]

        x = torch.reshape(y_val, (y_val.shape[0],1))
        x = x* F.linear(x_val, wg, bg) # / torch.linalg.norm(wg)

        x1 = torch.reshape(y_test, (y_test.shape[0],1)) 
        x1 = x1 * F.linear(x_test, wg, bg) # / torch.linalg.norm(wg)
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss = test_loss_upper.detach().numpy()/y_test.shape[0]

        loss_upper = val_loss
        loss_lower = (1/2) * (wg**2).sum()

        ###### Accuracy
        q = y_train * (wg @ x_train.T + bg)
        train_acc = (q>0).sum() / len(y_train)

        q = y_val * (wg @ x_val.T + bg)
        val_acc = (q>0).sum() / len(y_val)

        q = y_test * (wg @ x_test.T + bg)
        test_acc = (q>0).sum() / len(y_test)

        q = y_train * (wF @ x_train.T + bF)
        train_acc_F = (q>0).sum() / len(y_train)

        q = y_val * (wF @ x_val.T + bF)
        val_acc_F = (q>0).sum() / len(y_val)

        q = y_test * (wF @ x_test.T + bF)
        test_acc_F = (q>0).sum() / len(y_test)

        metrics.append({
            #'train_loss': train_loss,
            'train_acc': train_acc,
            'train_acc_F': train_acc_F,
            'val_loss': val_loss,
            'val_loss_F': val_loss_F,
            'val_acc': val_acc,
            'val_acc_F': val_acc_F,
            'test_loss': test_loss,
            'test_loss_F': test_loss_F,
            'test_acc': test_acc,
            'test_acc_F': test_acc_F,
            'loss_upper': loss_upper,
            'loss_lower': loss_lower,
            'time_computation': time.time()-algorithm_start_time
        })

        # Finding lower level variables and Lagrange Multipliers
        wg, bg, mug = minmax_opt(wg, bg, mug, lag_g, C_tensor_val, gam, eta1g, eta2g, T, Ty, y_val, x_val, y_train, x_train)
        wF, bF, muF = minmax_opt(wF, bF, muF, lag_F, C_tensor_val, gam, eta1F, eta2F, T, Ty, y_val, x_val, y_train, x_train)
        
        # Upper level iteration
        C_tensor_val.requires_grad_(True)

        x = torch.reshape(y_val, (y_val.shape[0],1)) 
        x = x* F.linear(x_val, wF, bF)
        loss_upper= torch.sum(torch.exp(1-x)) + torch.linalg.norm(C_tensor_val)

        C_tensor_val.grad = None # Reset gradients
        loss_upper.backward()

        ############# update on upper level variable C
        C_tensor_val.data -= eta*(C_tensor_val.grad.detach() + gam*mug - muF)
        C_tensor_val.data = torch.maximum(C_tensor_val.data, torch.tensor(1e-4))
        C_tensor_val.requires_grad_(False)
        
        #################
        if epoch%20==0 and verbose:
            print(f"Epoch [{epoch}/{epochs}]:",
              "val acc: {:.2f}".format(val_acc),
              "val loss: {:.2f}".format(val_loss),
              "test acc: {:.2f}".format(test_acc),
              "test loss: {:.2f}".format(test_loss))
            # print(f"Epoch [{j}/{epoch}]:","upper_loss: ", loss_upper.detach().numpy()/15.0, "test_loss_upper: ", test_loss_upper.detach().numpy()/11.8)

        val_loss_list.append(val_loss) # length 150
        test_loss_list.append(test_loss) # length 118
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        time_computation.append(time.time()-algorithm_start_time)

        if torch.linalg.norm(C_tensor_val.grad.detach() + gam*mug - muF) < early_stopping_th:
            break

    return metrics, variables


if __name__ == "__main__":
    ############ Load data code ###########

    data_utils = load_diabetes()

    data_list=[]

    f = open("../diabete.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()
    for line in a_list:
        line1=line.replace('\n', '')
        line2=list(line1.split(' '))
        y=float(line2[0])
        x= [float(line2[i].split(':')[1]) for i in (1,2,3,4,5,6,7,8)]
        data_list.append(x+[y])


    data_array_1=np.array(data_list)[:,:-1]
    data_array_0=np.ones((data_array_1.shape[0],1))
    data_array_2=data_array_1*data_array_1
    data_array_3=np.empty((data_array_1.shape[0],0))

    for i in range(data_array_1.shape[1]):
        for j in range(data_array_1.shape[1]):
            if i<j:
                data_array_i=data_array_1[:,i]*data_array_1[:,j]
                data_array_i=np.reshape(data_array_i,(-1,1))
                data_array_3=np.hstack((data_array_3,data_array_i))

    data_array_4=np.reshape(np.array(data_list)[:,-1],(-1,1))
    data=np.hstack((data_array_0,data_array_1,data_array_2,data_array_3,data_array_4))

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'gam': 5,
        'eta': 0.1
    }

    epochs = 80
    plot_results = True

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = ours(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
        metrics.append(metrics_seed)
        variables_seed.append(variables_seed)

    train_acc = np.array([[x['train_acc'] for x in metrics] for metrics in metrics])
    val_acc = np.array([[x['val_acc'] for x in metrics] for metrics in metrics])
    test_acc = np.array([[x['test_acc'] for x in metrics] for metrics in metrics])

    val_loss = np.array([[x['val_loss'] for x in metrics] for metrics in metrics])
    test_loss = np.array([[x['test_loss'] for x in metrics] for metrics in metrics])

    time_computation = np.array([[x['time_computation'] for x in metrics] for metrics in metrics])

    if plot_results:
        val_loss_mean=np.mean(val_loss,axis=0)
        val_loss_sd=np.std(val_loss,axis=0)/2.0
        test_loss_mean=np.mean(test_loss,axis=0)
        test_loss_sd=np.std(test_loss,axis=0)/2.0

        val_acc_mean=np.mean(val_acc,axis=0)
        val_acc_sd=np.std(val_acc,axis=0)/2.0
        test_acc_mean=np.mean(test_acc,axis=0)
        test_acc_sd=np.std(test_acc,axis=0)/2.0

        axis = np.mean(time_computation,axis=0)

        plt.rcParams.update({'font.size': 18})
        plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
        plt.rcParams['axes.unicode_minus']=False #显示负号
        axis=time_computation.mean(0)
        plt.figure(figsize=(8,6))
        #plt.grid(linestyle = "--") #设置背景网格线为虚线
        ax = plt.gca()
        plt.plot(axis,val_loss_mean,'-',label="Training loss")
        ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
        plt.plot(axis,test_loss_mean,'--',label="Test loss")
        ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        #plt.legend(loc=4)
        plt.ylabel("Loss")
        #plt.xlim(-0.5,3.5)
        #plt.ylim(0.5,1.0)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_1.pdf') 
        #plt.show()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.plot(axis,val_acc_mean,'-',label="Training accuracy")
        ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
        plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
        ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        plt.ylabel("Accuracy")
        # plt.ylim(0.64,0.8)
        #plt.legend(loc=4)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_2.pdf') 
        plt.show()
