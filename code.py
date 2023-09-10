import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import tensorflow_probability as tfp
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


sys.path.insert(0, r'...\Utilities')

np.random.seed(1204)
tf.random.set_seed(1204)


def lhs(d, N):
    x = np.linspace(0, 1, N)
    return np.array([x**i for i in range(d)]).T

class PhyInfNN:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1) 
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) 
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) 

        self.lb = lb
        self.ub = ub               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]
        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u0 = u0
        self.v0 = v0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Variables 
        self.x0_tf = tf.Variable(self.x0, dtype=tf.float32)
        self.t0_tf = tf.Variable(self.t0, dtype=tf.float32)
        self.u0_tf = tf.Variable(self.u0, dtype=tf.float32)
        self.v0_tf = tf.Variable(self.v0, dtype=tf.float32)
        self.x_lb_tf = tf.Variable(self.x_lb, dtype=tf.float32)
        self.t_lb_tf = tf.Variable(self.t_lb, dtype=tf.float32)
        self.x_ub_tf = tf.Variable(self.x_ub, dtype=tf.float32)
        self.t_ub_tf = tf.Variable(self.t_ub, dtype=tf.float32)
        self.x_f_tf = tf.Variable(self.x_f, dtype=tf.float32)
        self.t_f_tf = tf.Variable(self.t_f, dtype=tf.float32)

        # tf Functions (updated from variables computation)
        u0_v0_pred = self.net_uv(self.x0_tf, self.t0_tf)
        self.u0_pred, self.v0_pred = u0_v0_pred[:2]
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))

        self.optimizer_Adam = tf.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, t):
        X = tf.concat([x, t], axis=1)
        uv = self.neural_net(X, self.weights, self.biases)[:, :2]
        u, v = tf.unstack(uv, axis=1)
        
        epsilon = 1e-5  # Small value for finite difference

        u_x = (self.neural_net(tf.concat([x + epsilon, t], axis=1), self.weights, self.biases)[:, 0:1] - u) / epsilon
        v_x = (self.neural_net(tf.concat([x + epsilon, t], axis=1), self.weights, self.biases)[:, 1:2] - v) / epsilon

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)
        
        epsilon = 1e-5  # Small value for finite difference

        u_t = (self.net_uv(x, t + epsilon)[0] - u) / epsilon
        u_xx = (self.net_uv(x + epsilon, t)[2] - u_x) / epsilon
        v_t = (self.net_uv(x, t + epsilon)[1] - v) / epsilon
        v_xx = (self.net_uv(x + epsilon, t)[3] - v_x) / epsilon
        
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def callback(self, loss):
        print('Loss: ', loss)
        

    def train(self, nIter):
        
        tf_dict = { self.x0_tf.ref(): self.x0, self.t0_tf.ref(): self.t0,
                    self.u0_tf.ref(): self.u0, self.v0_tf.ref(): self.v0,
                    self.x_lb_tf.ref(): self.x_lb, self.t_lb_tf.ref(): self.t_lb,
                    self.x_ub_tf.ref(): self.x_ub, self.t_ub_tf.ref(): self.t_ub,
                    self.x_f_tf.ref(): self.x_f, self.t_f_tf.ref(): self.t_f }

        start_time = time.time()

        for it in range(nIter):
            with tf.GradientTape() as tape:
                loss_value = self.loss

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print(f'It: {it}, Loss: {loss_value.numpy()}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict(self, X_star):
        tf_dict = {self.x0_tf.ref(): X_star[:, 0:1], self.t0_tf.ref(): X_star[:, 1:2]}
        u_star, v_star = self.u0_pred.numpy(), self.v0_pred.numpy()

        tf_dict = {self.x_f_tf.ref(): X_star[:, 0:1], self.t_f_tf.ref(): X_star[:, 1:2]}
        f_u_star, f_v_star = self.f_u_pred.numpy(), self.f_v_pred.numpy()

        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]
    data = scipy.io.loadmat(r'...\NLS.mat')

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = u.T.flatten()[:,None]
    v_star = v.T.flatten()[:,None]
    h_star = h.T.flatten()[:,None]


    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f)

    model = PhyInfNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    start_time = time.time()
    model.train(100)
    elapsed = time.time() - start_time
    print('Training time: %.4f' %(elapsed))

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
  
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))


  ########################### PLOTTING ###############################
  
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])
    
    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    ########################## Row 0: h(t,x) ##################    
  
    gs0 = gridspec.GridSpec(2, 4)
    gs0.update(top=1-0.05, bottom=1-1.6/3, left=0.05, right=0.95, wspace=2)
    ax = plt.subplot(gs0[:, :])
    
    h_pred_reshaped = h_pred.reshape(1, -1)
    h = ax.imshow(h_pred_reshaped.T, interpolation='nearest', cmap='Blues', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$|h(t,x)|$', fontsize = 2)
    
       
################## Row 1: h(t,x) slices ##################   

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=0, bottom=-1, left=0.1, right=0.9, wspace=0.2)
  
    ax = plt.subplot(gs1[0, 0])
    x = x.flatten()
    ax.plot(x, Exact_h[:,75], 'b-', linewidth=1, label='Exact')       
    ax.plot(x, h_pred_reshaped[0,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=7)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

  ################## Row 2: h(t,x) slices ##################
  
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,h_pred_reshaped[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

################## Row 3: h(t,x) slices ##################
  
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,h_pred_reshaped[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
  
  #   savefig(r"...\figures")
