# practice LSTM
# https://github.com/nicodjimenez/lstm

#encoding:utf-8
'''
@ time t
lstm input:
1.Xt , current input node
2.Ht-1, last time output
3.Ct-1 , last time cell state

lstem output:
1. Ht, current output
2. Ct, current cell state


'''
import random
import numpy as np
import  math

def sigmoid(x):
    return 1./(1 + np.exp(-x))

# https://www.google.com/search?source=univ&tbm=isch&q=sigmoid+derivative&client=firefox-b-d&sa=X&ved=2ahUKEwj2i8LA2NjpAhXuyIsBHQsRDBMQsAR6BAgJEAE&biw=1290&bih=1195#imgrc=csorHz8n48WENM
def sigmoid_derivative(values):
    return values*(1-values)

def tanh_derivative(values):
    return  1. - values ** 2

# createst uniform random array w/ values in [a,b] and shape args
def rand_arr(a,b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

# model parameter  ,should be parameter of model
class LstmParam:
    # mem_cell_ct : cell state dim  ? , x_dim: inpput dim
    def __init__(self,mem_cell_ct,x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        cocat_len = x_dim + mem_cell_ct

        # weight matrices init
        self.wg = rand_arr(-0.1,0.1,mem_cell_ct,cocat_len) #  input node
        self.wi = rand_arr(-0.1,0.1,mem_cell_ct,cocat_len) # input gate
        self.wf = rand_arr(-0.1,0.1,mem_cell_ct,cocat_len) # forget date
        self.wo = rand_arr(-0.1,0.1,mem_cell_ct,cocat_len) #output weight

        #bias terms
        self.bg = rand_arr(-0.1,0.1,mem_cell_ct) #
        self.bi = rand_arr(-0.1,0.1,mem_cell_ct)
        self.bf = rand_arr(-0.1,0.1,mem_cell_ct)
        self.bo = rand_arr(-0.1,0.1,mem_cell_ct)

        # diffs (derivative of loss function w.r.t all parameters)
        self.wg_diff = np.zeros((mem_cell_ct,cocat_len))
        self.wi_diff = np.zeros((mem_cell_ct,cocat_len))
        self.wf_diff = np.zeros((mem_cell_ct,cocat_len))
        self.wo_diff = np.zeros((mem_cell_ct,cocat_len))
        self.bg_diff = np.zeros(mem_cell_ct) # you copy
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    # update weight
    def apply_diff(self,lr = 1):

        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bo -= lr * self.bo_diff
        # rest diff to zero  ,you copy
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

#? lstem cell state structure ?!
class LstmState:
    def __init__(self,mem_cell_ct,x_dim):
        # cell state , input x
        # ?????????
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct) # cell state ???!!!
        self.h = np.zeros(mem_cell_ct) # output?
        self.bottom_diff_h = np.zeros_like(self.h) #?
        self.bottom_diff_s = np.zeros_like(self.s)# ?

# lstm model node structure ?
class LstmNode:
    def __init__(self,lstm_param, lstm_state):
        # store reference to parameter and to activations
        self.state = lstm_state
        self.param = lstm_param

        # no recurrent input concatenated with recurrent input (xi + ht-1)?
        self.xc = None

    def bottom_data_is(self , x , s_prev = None , h_prev = None):
        # if this is the first lstm node in the network
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        #save data for use in backprop
        self.s_prev = s_prev # =======not define in init ?
        self.h_prev = h_prev

        #concatenate x(t) and h(t-1)
        # https://blog.csdn.net/u012609509/article/details/70319293
        xc = np.hstack((x,h_prev))
        # https://www.google.com/url?sa=i&url=https%3A%2F%2Flaoweizz.blogspot.com%2F2019%2F01%2Frecurrent2.html&psig=AOvVaw3LosPy3a5l0U7Za-y-itmB&ust=1590906949090000&source=images&cd=vfe&ved=0CAkQjhxqFwoTCMjIh7r82ukCFQAAAAAdAAAAABAI
        # 老尉子的部落格: Recurrent Neural Networks (RNN) 原理2/2
        self.state.g = np.tanh(np.dot(self.param.wg,xc) + self.param.bg) # input node
        self.state.i = sigmoid(np.dot(self.param.wi,xc) + self.param.bi) # input gate
        self.state.f = sigmoid(np.dot(self.param.wf,xc) + self.param.bf) # forget gate
        self.state.o = sigmoid(np.dot(self.param.wo,xc) + self.param.bo) # output gate
        self.state.s = s_prev * self.state.f + self.state.g * self.state.i #cell state t
        self.state.h = self.state.s * self.state.o #output t

        self.xc = xc


    # 往時間t-1 微分 傳回？！
    def top_diff_is(self,top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s #????
        do = self.state.s * top_diff_h
        di = self.state.g * ds # ????
        dg = self.state.i * ds #???
        df = self.s_prev * ds

        # differs w.r.t vector inisde sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        # https://www.itread01.com/content/1541623522.html
        self.param.wi_diff += np.outer(di_input,self.xc) #???? what
        self.param.wf_diff += np.outer(df_input,self.xc)
        self.param.wo_diff += np.outer(do_input,self.xc)
        self.param.wg_diff += np.outer(dg_input,self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        #compute bottom diff ???
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T,df_input)
        dxc += np.dot(self.param.wo.T,do_input)
        dxc += np.dot(self.param.wg.T,dg_input)

        #save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:] # ??

#Network structure
class LstmNetwork():

    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []

        #input sequence
        self.x_list = []

    # target list #----lstm node loss at t ?!
    def y_list_is(self, y_list, loss_layer):
        """
        Updates differs by setting target sequence
        with corresponding loss layer ???
        Will *NOT* update parameters.
        To update parameters, call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1 #iinput index

        #first node only get differs from label
        h = self.lstm_node_list[idx].state.h
        y = y_list[idx]
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # here s is not affecting loss due to h(t+1) , hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1  # from time sequence last one  to calculate

        ### following nodes also get diffs from next nodes, hence  we add diffs to diff_h
        ### we also propagate error along costant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h,y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx]) #???
            diff_h  += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss


    def x_list_clear(self):
        self.x_list = []


    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            #need to add new lstm node, create new state name
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)

        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)










