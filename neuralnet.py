__author__ = 'user'
import numpy as np
import sys
import math
import csv
import matplotlib.pyplot as plt
import pylab as py

def create_data_matrix(train_file):
    with open(train_file, 'rb') as f:
        reader = csv.reader(f)
        data = np.asarray(list(reader))

    labels=data[:,0]
    data=np.delete(data,0,1)
    return data,labels

def init_weight_matrix(hidden_units,init_flag, K, num_col):
    dimension_alpha=(hidden_units,num_col)
    dimension_beta=(K,hidden_units+1)

    if init_flag==1:
        # as bias terms has to be 0 so deleting first column and then appending zero column
        # also as 0.1 needs to be inclusive so taken limit 0.11 so that 0.1
        # becomes inclusive as high in exclusive in the function

        alpha=np.random.uniform(-0.1,0.1,dimension_alpha)
        alpha=np.delete(alpha,0,1)
        append_zero=np.zeros(hidden_units)
        alpha=np.hstack((append_zero[:, np.newaxis], alpha))

        beta=np.random.uniform(-0.1,0.1,dimension_beta)
        beta=np.delete(beta,0,1)
        append_zero=np.zeros(K)
        beta=np.hstack((append_zero[:, np.newaxis], beta))
    else:
        alpha=np.zeros(dimension_alpha)
        beta=np.zeros(dimension_beta)

    return alpha,beta

def linear_forward(alpha,feature):
    return np.dot(alpha,feature)

def sigmoid_forward(a):
    ones=np.ones(len(a))
    exp=np.exp(-1*a)
    sigma=np.divide(ones,(np.add(ones,exp)))
    return sigma

def softmax_forward(b):
    exp=np.exp(b)
    sum_exp=np.sum(exp)
    softmax=np.divide(exp,sum_exp)
    return softmax

def cross_entropy_forward(y,y_hat):
    loss=-1*(math.log(y_hat[y]))
    return loss

def cross_entropy_backward(y,y_hat):
    g_y_hat=np.divide(y,y_hat)
    return g_y_hat

def softmax_backward(g_y_hat,y_hat):
    g_y_hat_transpose=np.array([g_y_hat])    #np.array([arr]) gives 1d array in row vector form i.e. the true transpose
    diag_y_hat=np.diag(y_hat)
    y_hat_transpose=y_hat.reshape((-1,1))       #this is real y_hat. Just represented in column vector
    mult=np.dot(y_hat_transpose,np.array([y_hat]))   #np.array([arr]) gives 1d array in row vector form i.e. the true transpose
    sub=diag_y_hat-mult
    gb=np.dot(g_y_hat_transpose,sub)
    return gb

def linear_backward_gbeta(gb,z):
    gb=gb.transpose()
    z_tran=np.array([z])
    gbeta=np.dot(gb,z_tran)
    return gbeta

def linear_backward_gz(beta,gb):
    gb=gb.transpose()
    beta_trans=beta.transpose()
    gz=np.dot(beta_trans,gb)
    return gz

def sigmoid_backward(gz,z):
    gz=gz.transpose()
    one_minus=1-z
    z_one_minus=np.multiply(z,one_minus)
    ga=np.multiply(gz,z_one_minus)
    return ga

def cross_entropy(list_cross_entropy,epoch,train_data,train_labels,alpha,beta,data):
    sum_entropy=0.0
    samples=len(train_data)
    for index,features in enumerate(train_data):
        y=np.zeros(10)

        #ForwardPropagation
        features=[float(i) for i in features]   #converting string values to float for multiplication
        a=linear_forward(alpha,features)
        z=sigmoid_forward(a)
        z_bias=np.insert(z,0,1)
        b=linear_forward(beta,z_bias)
        y_hat=softmax_forward(b)
        J=cross_entropy_forward(int(train_labels[index]),y_hat)
        sum_entropy=sum_entropy+J

    avg_entropy=float(sum_entropy)/samples
    #s="epoch="+str(epoch+1)+" crossentropy("+data+"): "+str(avg_entropy)
    list_cross_entropy.append(avg_entropy)
    return

def training(list_cross_each_train,list_cross_each_test,train_data,train_labels,alpha,beta,test_data,test_labels,num_epoch,learn_rate):

    for epoch in xrange(0,num_epoch):
        for index,features in enumerate(train_data):
            y=np.zeros(10)

            #ForwardPropagation
            features=[float(i) for i in features]   #converting string values to float for multiplication
            #features=train_data.astype(float)
            a=linear_forward(alpha,features)
            z=sigmoid_forward(a)
            z_bias=np.insert(z,0,1)
            b=linear_forward(beta,z_bias)
            y_hat=softmax_forward(b)
            J=cross_entropy_forward(int(train_labels[index]),y_hat)
            #J=cross_entropy_forward(train_labels,y_hat)

            #BackPropagation
            y[int(train_labels[index])]=-1
            g_y_hat=cross_entropy_backward(y,y_hat)
            gb=softmax_backward(g_y_hat,y_hat)
            gbeta=linear_backward_gbeta(gb,z_bias)
            gz=linear_backward_gz(beta,gb)
            ga=sigmoid_backward(gz,z_bias)
            ga=np.delete(ga,0,1)
            galpha=linear_backward_gbeta(ga,features)

            #Weight update
            alpha=alpha-(learn_rate*(galpha))
            beta=beta-(learn_rate*(gbeta))

        #cross_entropy(list_cross_each_train,epoch,train_data,train_labels,alpha,beta,"train")
        #cross_entropy(list_cross_each_test,epoch,test_data,test_labels,alpha,beta,"test")
    return alpha,beta


def test(alpha,beta,data,labels,target_file):
    fw=open(target_file,"w")
    error_count=0
    for index,features in enumerate(data):
        y=np.zeros(10)

        #ForwardPropagation
        features=[float(i) for i in features]   #converting string values to float for multiplication
        a=linear_forward(alpha,features)
        z=sigmoid_forward(a)
        z_bias=np.insert(z,0,1)
        b=linear_forward(beta,z_bias)
        y_hat=softmax_forward(b)
        prediction=np.argmax(y_hat)

        fw.write("{}\n".format(prediction))
        if(prediction!=int(labels[index])):
            error_count=error_count+1
    fw.close()
    error_value=float(error_count)/(len(labels))
    return error_value

if __name__ == "__main__":


    train_data,train_labels=create_data_matrix(sys.argv[1])
    append_ones=np.ones(len(train_data),dtype=int)
    train_data=np.hstack((append_ones[:, np.newaxis], train_data))

    test_data,test_labels=create_data_matrix(sys.argv[2])
    append_ones=np.ones(len(test_data),dtype=int)
    test_data=np.hstack((append_ones[:, np.newaxis], test_data))

    alpha,beta=init_weight_matrix(int(sys.argv[7]), int(sys.argv[8]), len(np.unique(train_labels)), train_data.shape[1])

    list_cross_entropy_train=[]
    list_cross_entropy_test=[]

    hidden_units=[5,20,50,100,200]
    for unit in hidden_units:
        alpha,beta=init_weight_matrix(unit, 1, len(np.unique(train_labels)), train_data.shape[1])
        trained_alpha,trained_beta=training(list_cross_entropy_train,list_cross_entropy_test,train_data,train_labels,alpha,beta,test_data,test_labels,100,0.01)
        cross_entropy(list_cross_entropy_train,1,train_data,train_labels,trained_alpha,trained_beta,"train")
        cross_entropy(list_cross_entropy_test,1,test_data,test_labels,trained_alpha,trained_beta,"test")


    py.xlabel('Number of Hidden units')
    py.ylabel('Average cross entropy')
    py.plot(hidden_units,list_cross_entropy_train,'r--',label='train data')
    py.plot(hidden_units,list_cross_entropy_test,'g--',label='test data')
    py.grid(True)
    py.legend(loc="upper right")
    py.show()


    #trained_alpha,trained_beta=training(list_cross_entropy,train_data,train_labels,alpha,beta,test_data,test_labels,int(sys.argv[6]),float(1))

    #train_error=test(trained_alpha,trained_beta,train_data,train_labels,sys.argv[3])
    #test_error=test(trained_alpha,trained_beta,test_data,test_labels,sys.argv[4])

    #str_train="error(train): "+str(train_error)
    #list_cross_entropy.append(str_train)
    #str_test="error(test): "+str(test_error)
    #list_cross_entropy.append(str_test)

    #with open(sys.argv[5], 'w') as f:
    #    for item in list_cross_entropy:
    #        f.write("%s\n" % item)






