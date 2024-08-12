import cvxpy as cp
import numpy as np
import time
import copy

import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def compute_accuracy(loader, model, model_theta):

    number_right = 0
    loss = 0
    loss_theta = 0
    number_right_theta = 0
    for batch_idx, (images, labels) in enumerate(loader): #val_loader
        images, labels = images.to(device), labels.to(device)
        log_probs = model(images)
        log_probs_theta = model_theta(images)
        for i in range(len(labels)):
            q=log_probs[i]*labels[i]
            if q>0:
                number_right=number_right+1
            q_theta = log_probs_theta[i]*labels[i]
            if q_theta>0:
                number_right_theta += 1
        loss += model.loss_upper(log_probs, labels)
        loss_theta += model_theta.loss_upper(log_probs, labels)
    acc=number_right/len(loader.dataset)
    acc_theta=number_right_theta/len(loader.dataset)
    loss /= len(loader.dataset)
    loss_theta /= len(loader.dataset)

    return loss, loss_theta, acc, acc_theta

class LinearSVM(nn.Module):
    def __init__(self, input_size, n_classes, n_sample):
        super(LinearSVM, self).__init__()
        self.w = nn.Parameter(torch.ones(n_classes, input_size))
        self.b = nn.Parameter(torch.tensor(0.))
        self.xi = nn.Parameter(torch.ones(n_sample))
        #self.C = nn.Parameter(10.*torch.ones(n_sample))
        self.C = nn.Parameter(torch.empty(n_sample))
        self.C.data.uniform_(1.,5.)
    
    def forward(self, x):
        return F.linear(x, self.w, self.b)

    def loss_upper(self, y_pred, y_val):
        y_val_tensor = torch.Tensor(y_val)
        x = torch.reshape(y_val_tensor, (y_val_tensor.shape[0],1)) * y_pred / torch.linalg.norm(self.w)
        # relu = nn.LeakyReLU()
        # loss = torch.sum(relu(2*torch.sigmoid(-5.0*x)-1.0))
        loss= torch.sum(torch.exp(1-x))
        # loss += 0.5*torch.linalg.norm(self.C)**2
        return loss

    def loss_lower(self):
        w2 = 0.5*torch.linalg.norm(self.w)**2
        #c_exp=torch.exp(self.C)
        #xi_term = 0.5 * (torch.dot(c_exp, (self.xi)**2))
        loss =  w2# + xi_term
        loss += 0.5*torch.linalg.norm(self.C)**2
        return loss

    def constrain_values(self, srt_id, y_pred, y_train):
        xi_sidx = srt_id
        xi_eidx = srt_id+len(y_pred)
        xi_batch = self.xi[xi_sidx:xi_eidx]
        return 1-xi_batch-y_train.view(-1)*y_pred.view(-1)

    def second_constraint_val(self):
        return self.xi - self.C

def lv_hba(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, compute_opt=False, early_stopping_th = False, verbose=True):

    batch_size = 256
    data_train = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True)
    data_val = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=batch_size,
        shuffle=True)
    data_test = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True)

    ############ Setting SVM ###########
    feature = 8
    feature=x_train.shape[1]
    N_sample = y_train.shape[0]
    
    model = LinearSVM(feature, 1, N_sample).to(device)
    model.C.data.copy_(torch.Tensor(x_train.shape[0]).uniform_(1.0,5.0))    ####### Setting C on training data
    model_theta = copy.deepcopy(model)    ####### ?????????????

    ######### SVM variables
    lamda = torch.ones(2*N_sample) #+ 1./N_sample
    z = torch.ones(2*N_sample) #+ 1./N_sample

    params = [p for n, p in model.named_parameters() if n != 'C']
    params_theta = [p for n, p in model_theta.named_parameters() if n != 'C']

    ############### Projection
    x = cp.Variable(feature+1+2*N_sample)
    y = cp.Parameter(feature+1+2*N_sample)
    w = x[0:feature]
    b = x[feature]
    xi = x[feature+1:feature+1+N_sample]
    C = x[feature+1+N_sample:]

    loss = cp.norm(x-y, 2)**2

    constraints=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) <= 0)
        constraints.append(xi[i] <= C[i])

    obj = cp.Minimize(loss)
    prob = cp.Problem(obj, constraints)
    ############### The above is for projection

    ############ LV-HBA parameter ###########   
    # alpha = 0.01
    # beta = 0.1
    # yita = 0.001
    # gama1 = 0.1
    # gama2 = 0.1
    # #ck = 0.1
    # u = 200

    # alpha = 0.01
    # beta = 0.1
    # beta = 0.01 
    # gama1 = 0.1
    # gama2 = 0.1
    # #ck = 0.1

    alpha = hparams['alpha']
    yita = hparams['yita']
    gama1 = hparams['gama1']
    gama2 = hparams['gama2']
    
    #epochs = 80
    algorithm_start_time=time.time()

    variables = []
    metrics = []

    if compute_opt:
        C_opt = cp.Parameter(y_train.shape[0], nonneg=True)
    
        w_opt = cp.Variable(feature)
        b_opt = cp.Variable()
        xi_opt = cp.Variable(y_train.shape[0], nonneg=True)

        loss_lower =  0.5*cp.norm(w_opt, 2)**2

        constraints=[]
        for i in range(y_train.shape[0]):
            constraints.append(1 - xi_opt[i] - y_train[i] * (cp.scalar_product(w_opt, x_train[i])+b_opt) <= 0)
        
        constraints_xi = [xi_opt <= C_opt]

        obj_lower = cp.Minimize(loss_lower)

        prob_lower = cp.Problem(obj_lower, constraints + constraints_xi)
    
    ############ LV-HBA algorithm ###########    
    for epoch in range(epochs):
        #### Storage of variables and metrics

        vars_to_append = {
            'C': model.C.data.clone(),
            'xi': model.xi.data.clone(),
            'w': model.w.data.clone(),
            'b': model.b.data.clone(),
            'lambda': lamda.clone(),
            'z': z.clone(),
            'xi_theta': model_theta.xi.data.clone(),
            'w_theta': model_theta.w.data.clone(),
            'b_theta': model_theta.b.data.clone()
        }

        if compute_opt:
            C_opt.value = model.C.data.view(-1).detach().numpy()
            prob_lower.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)
            vars_to_append['xi_opt'] = torch.tensor(xi_opt.value)
            vars_to_append['w_opt'] = torch.tensor(w_opt.value)
            vars_to_append['b_opt'] = torch.tensor(b_opt.value)

        variables.append(vars_to_append)

        with torch.no_grad():

            train_loss, train_loss_theta, train_acc, train_acc_theta = compute_accuracy(train_loader, model, model_theta)
            val_loss, val_loss_theta, val_acc, val_acc_theta = compute_accuracy(val_loader, model, model_theta)
            test_loss, test_loss_theta, test_acc, test_acc_theta = compute_accuracy(test_loader, model, model_theta)

            # Need to recompute to prevent norm of c tensor
            x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
            x = x* F.linear(torch.Tensor(x_val), model.w.data, model.b.data) # / torch.linalg.norm(w_tensor)
            val_loss= torch.sum(torch.exp(1-x))/len(val_loader.dataset)
            x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
            x1 = x1 * F.linear(torch.Tensor(x_test), model.w.data, model.b.data) # / torch.linalg.norm(w_tensor)
            test_loss= torch.sum(torch.exp(1-x1))/len(test_loader.dataset)

        metrics.append({
            'train_loss': train_loss,
            'train_loss_theta': train_loss_theta.detach().numpy(),
            'train_acc': train_acc,
            'train_acc_theta': train_acc_theta,
            'val_loss': val_loss,
            'val_loss_theta': val_loss_theta.detach().numpy(),
            'val_acc': val_acc,
            'val_acc_theta': val_acc_theta,
            'test_loss': test_loss,
            'test_loss_theta': test_loss_theta.detach().numpy(),
            'test_acc': test_acc,
            'test_acc_theta': test_acc_theta,
            'loss_lower': 0.5*torch.linalg.norm(model.w.data)**2,
            'time_computation': time.time()-algorithm_start_time
        })

        ### Start of the algorithm
        ck = 1/((epoch+1)**0.5)

        ################## Lower Level
        model_theta.zero_grad()
        loss = model_theta.loss_lower()
        loss.backward()

        ############### go through training to build up idx_glob constr_glob_list
        idx_glob = 0
        constr_glob_list = torch.ones(0)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model_theta(images)
            cv = model_theta.constrain_values(idx_glob, log_probs, labels)
            lamda_batch = lamda[idx_glob:idx_glob+len(labels)]
            cv.backward(lamda_batch)
            constr_glob_list = torch.cat((constr_glob_list, cv), 0)
            idx_glob += len(labels)
        # Constraint xi <= C
        cv = model_theta.second_constraint_val()
        cv.backward(lamda[N_sample:])
        constr_glob_list = torch.cat((constr_glob_list, cv), 0)

        # ############ This is the algorithm
        for i, p_theta in enumerate(params_theta):
            d4_theta = torch.zeros_like(p_theta.data)
            if p_theta.grad is not None:
                d4_theta += p_theta.grad
            d4_theta += gama1*(p_theta.data - params[i].data)
            p_theta.data.add_(d4_theta, alpha=-yita)

        lamda = lamda - yita*(-constr_glob_list + gama2*(lamda - z))

        ############## go through training again to build up idx_glob in another order
        model_theta.zero_grad()
        loss = model_theta.loss_lower()
        loss.backward()

        idx_glob = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model_theta(images)
            cv = model_theta.constrain_values(idx_glob, log_probs, labels)
            lamda_batch = lamda[idx_glob:idx_glob+len(labels)]
            cv.backward(lamda_batch)
            idx_glob += len(labels)
        # Constraint xi <= C
        cv = model_theta.second_constraint_val()
        cv.backward(lamda[N_sample:])


        # ############## upper using model on validation data
        model.zero_grad()
        loss = model.loss_lower() #loss_upper
        loss.backward()

        loss_upper = 0
        ################ Upper lever: validation data
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            loss = model.loss_upper(log_probs, labels)
            loss_upper += loss
            loss.backward(torch.tensor(ck))
        
        for i, p in enumerate(params):
            d2 = torch.zeros_like(p.data)
            if p.grad is not None:
                d2 += p.grad
            d2 += gama1*(params_theta[i].data - p.data)
            p.data.add_(d2, alpha=-alpha)

        d1 = model.C.grad - model_theta.C.grad
        model.C.data.add(d1, alpha=-alpha)

        
        #prob.solve(solver='MOSEK', warm_start=True, verbose=True)
        y_w = model.w.data.view(-1).detach().numpy()
        y_b = model.b.data.detach()
        y_xi = model.xi.data.view(-1).detach().numpy()
        y_C = model.C.data.view(-1).detach().numpy()

        y.value = np.concatenate((y_w, np.array([y_b]), y_xi, y_C))

        #### ?????????? projection
        prob.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  
        C_solv = torch.tensor(np.array(C.value))
        w_solv = torch.tensor(np.array([w.value]))
        b_solv = torch.tensor(b.value)
        xi_solv = torch.tensor(np.array(xi.value))

        model_theta.C.data.copy_(C_solv)
        model.C.data.copy_(C_solv)
        model.w.data.copy_(w_solv)
        model.b.data.copy_(b_solv)
        model.xi.data.copy_(xi_solv)

        if epoch%20==0 and verbose:
            print("val acc: {:.2f}".format(val_acc),
              "val loss: {:.2f}".format(val_loss),
              "test acc: {:.2f}".format(test_acc),
              "test loss: {:.2f}".format(test_loss),
              "round: {}".format(epoch))
            
        if torch.linalg.norm(d1) < early_stopping_th:
            break

    return metrics, variables


if __name__ == "__main__":
    ############ Load data code ###########

    data = load_diabetes()

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
        'alpha': 0.01,
        'gama1': 0.1,
        'gama2': 0.1,
        'yita': 0.001
    }

    epochs = 80
    plot_results = True

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = lv_hba(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
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
