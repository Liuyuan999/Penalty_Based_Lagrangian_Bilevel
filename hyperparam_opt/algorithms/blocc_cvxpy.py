import cvxpy as cp
import numpy as np
import time

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split


def blocc(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, verbose=True):
    feature=x_train.shape[1] # = 8
    ######### parameters
    C_tensor_val= torch.Tensor(x_train.shape[0]).uniform_(1.,5.)

    ###### Ours paramter
    eta = hparams['eta']
    gam = hparams['gam']

    C = cp.Parameter(y_train.shape[0], nonneg=True)
    
    # Parameters for eq. (12)
    w = cp.Variable(feature)
    b = cp.Variable()
    xi = cp.Variable(y_train.shape[0], nonneg=True)
    

    # Parameters for eq. (13)
    w_F = cp.Variable(feature)
    b_F = cp.Variable()
    xi_F = cp.Variable(y_train.shape[0], nonneg=True)

    ######### 2 level objectives
    loss_lower =  0.5*cp.norm(w, 2)**2# + 0.5 * (cp.scalar_product(C, cp.power(xi,2))) # cp.exp(C)
        
    # Compute the final expression
    
    loss_lower_F =  0.5*cp.norm(w_F, 2)**2# + 0.5  * (cp.scalar_product( C,  cp.power(xi_F,2)))
    loss_upper =  cp.sum(
        # cp.maximum(
        #     0, 1- cp.multiply(y_val, x_val@w_F + b_F)
        # )**2
        cp.exp( 1 - cp.multiply(y_val, x_val@w_F + b_F) )
    )

    # Create two constraints.
    constraints=[]
    constraints_value=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) <= 0)
        constraints_value.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) )
    
    constraints_xi = [xi <= C]

    constraints_F=[]
    constraints_value_F=[]
    for i in range(y_train.shape[0]):
        constraints_F.append(1 - xi_F[i] - y_train[i] * (cp.scalar_product(w_F, x_train[i])+b_F) <= 0)
        constraints_value_F.append(1 - xi_F[i] - y_train[i] * (cp.scalar_product(w_F, x_train[i])+b_F) )

    constraints_xi_F = [xi_F <= C]

    # Form objective.
    obj_lower = cp.Minimize(loss_lower)

    obj_F = cp.Minimize(loss_lower_F + 1/gam*loss_upper)

    # Form and solve problem.
    prob_lower = cp.Problem(obj_lower, constraints + constraints_xi)
    prob_F = cp.Problem(obj_F, constraints_F + constraints_xi_F)

    w_tensor = torch.ones(1,feature)
    b_tensor = torch.tensor(0.)
    xi_tensor = torch.tensor(y_train.shape[0])
    w_F_tensor = w_tensor.clone()
    b_F_tensor = b_tensor.clone()
    xi_F_tensor = xi_tensor.clone()
    
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
            'xi': xi_tensor,
            'w': w_tensor,
            'b': b_tensor,
            'xi_F': xi_F_tensor,
            'w_F': w_F_tensor,
            'b_F': b_F_tensor
        })

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)

        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        x1 = x1 * F.linear(torch.Tensor(x_test), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)
        # test_loss_upper= torch.sum(torch.sigmoid(x1))
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss_F = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss_F = test_loss_upper.detach().numpy()/y_test.shape[0]

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)

        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        x1 = x1 * F.linear(torch.Tensor(x_test), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        # test_loss_upper= torch.sum(torch.sigmoid(x1))
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss = test_loss_upper.detach().numpy()/y_test.shape[0]

        ###### Accuracy
        q = torch.tensor(y_train) * (w_tensor @ x_train.T + b_tensor)
        train_acc = (q>0).sum() / len(y_train)

        q = torch.tensor(y_val) * (w_tensor @ x_val.T + b_tensor)
        val_acc = (q>0).sum() / len(y_val)

        q = torch.tensor(y_test) * (w_tensor @ x_test.T + b_tensor)
        test_acc = (q>0).sum() / len(y_test)

        q = torch.tensor(y_train) * (w_F_tensor @ x_train.T + b_F_tensor)
        train_acc_F = (q>0).sum() / len(y_train)

        q = torch.tensor(y_val) * (w_F_tensor @ x_val.T + b_F_tensor)
        val_acc_F = (q>0).sum() / len(y_val)

        q = torch.tensor(y_test) * (w_F_tensor @ x_test.T + b_F_tensor)
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
            'loss_lower': prob_lower.value,
            'time_computation': time.time()-algorithm_start_time
        })

        c_array_value_np = C_tensor_val.detach().numpy()# /c_array_tensor.detach().numpy().sum()
        # print(c_array_value_np.sum(),sum(c_array))
        C.value = c_array_value_np

        ###### Solve Eq.(12), (13)
        begin=time.time()
        try:
            prob_lower.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  
            prob_F.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)
        except:
            print(C.value)
            print(prob_lower.status)
            print(prob_F.status)
            prob_lower.solve(solver='SCS')  
            prob_F.solve(solver='SCS')
            raise RuntimeError("Lo he resuelto")
        end=time.time()
        # print("time: ",end-begin)

        dual_variables = np.array([ constraints[i].dual_value for i in range(len(constraints))])
        constraints_value_1= np.array([ constraints_value[i].value for i in range(len(constraints))])
        dual_variables_xi = constraints_xi[0].dual_value
        
        dual_variables_F = np.array([ constraints_F[i].dual_value for i in range(len(constraints_F))])
        constraints_value_1_F= np.array([ constraints_value_F[i].value for i in range(len(constraints_F))])
        dual_variables_xi_F = constraints_xi_F[0].dual_value

        ############# Calculate gradient
        try:
            w_tensor=torch.Tensor(np.array([w.value])) #.requires_grad_()
        except:
            print(prob_lower.status)
            print(prob_F.status)
            print(w_tensor)
            raise RuntimeError("HE DADO NONE")
        b_tensor=torch.Tensor(np.array([b.value])) #.requires_grad_()
        xi_tensor =torch.Tensor(np.array([xi.value]))
        C_tensor=torch.Tensor(np.array([C.value])).requires_grad_()
                
        ############# Calculate gradient 
        w_F_tensor=torch.Tensor(np.array([w_F.value])) #.requires_grad_()
        b_F_tensor=torch.Tensor(np.array([b_F.value])) #.requires_grad_()
        xi_F_tensor =torch.Tensor(np.array([xi_F.value]))

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)
        loss_upper= torch.sum(torch.exp(1-x)) + torch.linalg.norm(C_tensor)

        loss_upper.backward()

        ############# update on upper level variable C
        C_tensor_val = C_tensor.detach()
        C_tensor_val -= eta*(C_tensor.grad.detach() + gam*dual_variables_xi - gam*dual_variables_xi_F) #the second gam* is due to the 1/gam in obj_F
        #C_tensor_val -= eta*(gam*dual_variables_xi) - dual_variables_xi_F
        C_tensor_val = torch.maximum(C_tensor_val, torch.tensor(1e-4))[0,:]
        
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

    return metrics, variables


if __name__ == "__main__":
    ############ Load data code ###########

    data = load_diabetes()

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'gam': 5,
        'eta': 0.1
    }

    epochs = 80
    plot_results = False

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = blocc(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
        metrics.append(metrics_seed)
        variables_seed.append(variables_seed)

    train_acc = np.array([[x['train_acc'] for x in metric] for metric in metrics])
    val_acc = np.array([[x['val_acc'] for x in metric] for metric in metrics])
    test_acc = np.array([[x['test_acc'] for x in metric] for metric in metrics])

    val_loss = np.array([[x['val_loss'] for x in metric] for metric in metrics])
    test_loss = np.array([[x['test_loss'] for x in metric] for metric in metrics])

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
