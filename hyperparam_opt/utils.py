import numpy as np

def load_fourclass():
    data_list=[]

    f = open("datasets/fourclass.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()
    for line in a_list:
        line1=line.replace('\n', '')
        line2=list(line1.split(' '))
        y=float(line2[0])
        x= [float(line2[i].split(':')[1]) if line2[i] != '' else 0 for i in (1,2)]
        data_list.append(x+[y])

    return np.array(data_list)

def load_diabetes():
    f = open("datasets/diabete.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()

    data_list = []
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

    return data

def train_val_test_split(data, seed, n_train, n_val):
    np.random.seed(seed)
    np.random.shuffle(data)

    # Assuming target variable being in the last column
    x_train=data[:n_train, :-1]
    y_train=data[:n_train, -1]
    x_val=data[n_train:n_train+n_val, :-1]
    y_val=data[n_train:n_train+n_val, -1]
    x_test=data[n_train+n_val:, :-1]
    y_test=data[n_train+n_val:, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test