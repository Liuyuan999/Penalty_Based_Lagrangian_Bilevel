import cvxpy as cp
import numpy as np
import time

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split


def gam(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, early_stopping_th = False, verbose=True):

    c_array= torch.Tensor(x_train.shape[0]).uniform_(-7.0,-6.0)
    #c_array= torch.Tensor(x_train.shape[0]).uniform_(1.0,2.0)
    c_array_tensor=torch.exp(c_array)

    feature=x_train.shape[1]

    w = cp.Variable(feature)
    b = cp.Variable()
    xi = cp.Variable(y_train.shape[0])
    C = cp.Parameter(y_train.shape[0],nonneg=True)
    loss =  0.5*cp.norm(w, 2)**2 + 0.5 * (cp.scalar_product(C, cp.power(xi,2)))

    # Create two constraints.
    constraints=[]
    constraints_value=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) <= 0)
        constraints_value.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) )

    # Form objective.
    obj = cp.Minimize(loss)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    w_tensor = torch.ones(1,feature)
    b_tensor = torch.tensor(0.)
    xi_tensor = torch.ones(y_train.shape[0])

    alpha = hparams['alpha']
    epsilon = hparams['epsilon']

    algorithm_start_time=time.time()

    variables = []
    metrics = []

    for epoch in range(epochs):
        C.value=c_array_tensor.detach().numpy()

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        loss_upper= torch.sum(torch.exp(1-x))
        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        x1 = x1 * F.linear(torch.Tensor(x_test), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss = loss_upper.detach().numpy()/y_val.shape[0]
        test_loss = test_loss_upper.detach().numpy()/y_test.shape[0]

        ######### Accuracy
        q = torch.tensor(y_train) * (w_tensor @ x_train.T + b_tensor)
        train_acc = (q>0).sum() / len(y_train)

        q = torch.tensor(y_val) * (w_tensor @ x_val.T + b_tensor)
        val_acc = (q>0).sum() / len(y_val)

        q = torch.tensor(y_test) * (w_tensor @ x_test.T + b_tensor)
        test_acc = (q>0).sum() / len(y_test)

        variables.append({
            'C': c_array_tensor,
            'xi': xi_tensor,
            'w': w_tensor.detach(),
            'b': b_tensor.detach()
        })

        metrics.append({
            'train_acc': train_acc,
            #'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'loss_lower': 0.5*torch.linalg.norm(w_tensor.detach())**2.,
            'time_computation': time.time() - algorithm_start_time
        })


        begin=time.time()
        prob.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  
        end=time.time()

        dual_variables= np.array([ constraints[i].dual_value for i in range(len(constraints))])
        constraints_value_1= np.array([ constraints_value[i].value for i in range(len(constraints))])

        w_tensor=torch.Tensor(np.array([w.value])).requires_grad_()
        b_tensor=torch.Tensor(np.array([b.value])).requires_grad_()
        xi_tensor = torch.Tensor(xi.value)

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        loss_upper= torch.sum(torch.exp(1-x))

        inactive_constraint_list=[]
        for i in range(len(y_train)):
            if constraints_value_1[i]<-0.00001:
                inactive_constraint_list.append(i)

        active_constraint_list=[]
        for i in range(len(y_train)):
            if dual_variables[i]>0.00001:
                active_constraint_list.append(i)

        #M = np.zeros((feature+1+y_train.shape[0]+len(active_constraint_list),feature+1+y_train.shape[0]+len(active_constraint_list)), dtype = float) 

        v1=np.ones((feature,))
        v2=np.zeros((1,))
        v3=c_array_tensor.detach().numpy()
        M1= np.diag(np.hstack((v1,v2,v3)))
        M2 = np.empty([0,0], dtype = float) 
        #v4= np.zeros((1, feature+1+y_train.shape[0]+len(active_constraint_list) ), dtype = float) 
        M2_list=[]
        for i in range(y_train.shape[0]):
            if i in active_constraint_list:
                M2_list.append( np.array([ np.hstack((x_train[i]* (-y_train[i]),np.array([-y_train[i]]),-np.eye(y_train.shape[0])[i])) ]) )
        M2= np.vstack(M2_list)

        M3= np.transpose(M2)
        M4 = np.zeros((len(active_constraint_list),len(active_constraint_list)))
        M = np.hstack((np.vstack((M1,M2)), np.vstack((M3,M4))))
        #print(M.shape)
        #print(np.linalg.matrix_rank(M))
        
        n1=np.zeros((feature+1, y_train.shape[0]))
        n2=np.diag(np.array(xi.value)*c_array_tensor.detach().numpy())
        n3=np.zeros((len(active_constraint_list),y_train.shape[0]))
        N=np.vstack((n1,n2,n3))
        #print(N.shape)

        d=-np.dot(np.linalg.inv(M), N) 
        d1=d[0:feature+1,]
        d2=d[feature+1:feature+1+y_train.shape[0],]
        d3=d[feature+1+y_train.shape[0]:feature+1+y_train.shape[0]+len(active_constraint_list),]

        loss_upper.backward()
        grads_w = w_tensor.grad.detach().numpy()[0]
        grads_b = b_tensor.grad.detach().numpy()
        grad=np.hstack((grads_w,grads_b))
        grad=np.reshape(grad,(1,grad.shape[0]))
        grad_update=np.dot(grad,d1)[0]
        c_array=c_array-alpha*grad_update
        c_array_tensor=torch.exp(c_array)

        w_tensor = w_tensor.detach()
        b_tensor = b_tensor.detach()

        if epoch%20==0 and verbose:
            print("val acc: {:.2f}".format(val_acc),
              "val loss: {:.2f}".format(val_loss),
              "test acc: {:.2f}".format(test_acc),
              "test loss: {:.2f}".format(test_loss),
              "round: {}".format(epoch))
        
        if np.linalg.norm(grad_update) < early_stopping_th:
            break
    return metrics,variables


if __name__ == "__main__":
    ############ Load data code ###########

    data = load_diabetes()

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'alpha': 0.05,
        'epsilon': 0.005
    }

    epochs = 80
    plot_results = True

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = gam(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
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
        plt.rcParams['font.sans-serif']=['Arial']
        plt.rcParams['axes.unicode_minus']=False

        axis=time_computation.mean(0)

        plt.figure(figsize=(8,6))
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
        plt.savefig('ho_svm_kernel_1.pdf') 
        plt.show()

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
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.savefig('ho_svm_kernel_2.pdf') 
        plt.show()
