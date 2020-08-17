import numpy as np
import matplotlib.pyplot as plt
import math
import random
import tensorflow as tf

seed = 42
random.seed(seed)


def random_mini_batches(XE, R1E, R2E, mini_batch_size = 10, seed = 42):
    # Creating the mini-batches
    np.random.seed(seed)
    m = int(XE.shape[0])
    mini_batches = []
    permutation = np.random.permutation(m)
    permutation = list(permutation)

    shuffled_XE = XE[permutation,:]
    shuffled_X1R = R1E[permutation]
    shuffled_X2R = R2E[permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_X1R = shuffled_X1R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X2R = shuffled_X2R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
        mini_batch_X1R = shuffled_X1R[Lower : Lower + Upper]
        mini_batch_X2R = shuffled_X2R[Lower : Lower + Upper]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)

    return mini_batches

N = 10000
M = 100
c = 0.5
p = 0.9
k = np.random.randn(M)
u1 = np.random.randn(M)
u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
u1 /= np.linalg.norm(u1)
k /= np.linalg.norm(k)
u2 = k
w1 = c*u1
w2 = c*(p*u1+np.sqrt((1-p**2))*u2)
X = np.random.normal(0, 1, (N, M))
eps1 = np.random.normal(0, 0.01)
eps2 = np.random.normal(0, 0.01)
Y1 = np.matmul(X, w1) + np.sin(np.matmul(X, w1))+eps1
Y2 = np.matmul(X, w2) + np.sin(np.matmul(X, w2))+eps2
split = list(np.random.permutation(N))

X_train = X[split[0:8000],:]
Y1_train = Y1[split[0:8000]]
Y2_train = Y2[split[0:8000]]
X_valid = X[8000:9000,:]
Y1_valid = Y1[8000:9000]
Y2_valid = Y2[8000:9000]
X_test = X[9000:10000,:]
Y1_test = Y1[9000:10000]
Y2_test = Y2[9000:10000]

# TensorFlow model
input_size, feature_size = X.shape
shared_layer_size = 64
tower_h1 = 32
tower_h2 = 16
output_size = 1
LR = 0.001
epoch = 50
mb_size = 100
cost1tr = []
cost2tr = []
costtr = []
cost1_val = []
cost2_val = []
cost_val = []
alpha = 0.12

l1_scale = 1.0
l2_scale = 100.0

# Model input
X_ = tf.placeholder("float", [None, feature_size], name="X")
Y1_ = tf.placeholder("float", [None, output_size], name="Y1")
Y2_ = tf.placeholder("float", [None, output_size], name="Y2")
keep_prob = tf.placeholder_with_default(1.0, shape=())

def linear_layer(in_size, out_size):
    k = tf.sqrt(1/float(in_size))
    a = tf.Variable(tf.random_uniform([in_size, out_size], minval=-k, maxval=k))
    b = tf.Variable(tf.random_uniform([out_size], minval=-k, maxval=k))
    return a, b

with tf.variable_scope("shared_layer"):
    # Linear layer with ReLU and droput
    weights_shared, bias_shared = linear_layer(feature_size, shared_layer_size)
    hidden_output_shared = tf.nn.relu(tf.matmul(tf.expand_dims(X_, 0), weights_shared) + bias_shared)
    hidden_output_shared = tf.nn.dropout(hidden_output_shared, rate=1-keep_prob)

with tf.variable_scope("tower1"):
    # Linear layer with ReLU and droput
    weights_t1_1, bias_t1_1 = linear_layer(shared_layer_size, tower_h1)
    hidden_output_t1_1 = tf.nn.relu(tf.matmul(hidden_output_shared, weights_t1_1) + bias_t1_1)
    hidden_output_t1_1 = tf.nn.dropout(hidden_output_t1_1, rate=1-keep_prob)

    # Linear layer with ReLU and droput
    weights_t1_2, bias_t1_2 = linear_layer(tower_h1, tower_h2)
    hidden_output_t1_2 = tf.nn.relu(tf.matmul(hidden_output_t1_1, weights_t1_2) + bias_t1_2)
    hidden_output_t1_2 = tf.nn.dropout(hidden_output_t1_2, rate=1-keep_prob)

    # Output linear layer
    weights_t1_3, bias_t1_3 = linear_layer(tower_h2, output_size)
    y1_out = tf.multiply(tf.reshape(tf.matmul(hidden_output_t1_2, weights_t1_3) + bias_t1_3, [-1, 1]), tf.constant(l1_scale))

with tf.variable_scope("tower2"):
    # Linear layer with ReLU and droput
    weights_t2_1, bias_t2_1 = linear_layer(shared_layer_size, tower_h1)
    hidden_output_t2_1 = tf.nn.relu(tf.matmul(hidden_output_shared, weights_t2_1) + bias_t2_1)
    hidden_output_t2_1 = tf.nn.dropout(hidden_output_t2_1, rate=1-keep_prob)

    # Linear layer with ReLU and droput
    weights_t2_2, bias_t2_2 = linear_layer(tower_h1, tower_h2)
    hidden_output_t2_2 = tf.nn.relu(tf.matmul(hidden_output_t2_1, weights_t2_2) + bias_t2_2)
    hidden_output_t2_2 = tf.nn.dropout(hidden_output_t2_2, rate=1-keep_prob)

    # Output linear layer
    weights_t2_3, bias_t2_3 = linear_layer(tower_h2, output_size)
    y2_out = tf.multiply(tf.reshape(tf.matmul(hidden_output_t2_2, weights_t2_3) + bias_t2_3, [-1, 1]), tf.constant(l2_scale))

with tf.variable_scope("loss_gradnorm"):
    # Task weights
    w1 = tf.Variable(1.0)
    w2 = tf.Variable(1.0)

    # Define loss functions
    loss_t1 = tf.losses.mean_squared_error(y1_out, Y1_)
    loss_t2 = tf.losses.mean_squared_error(y2_out, Y2_)

    # Scale loss using weights
    l1 = tf.multiply(loss_t1, w1)
    l2 = tf.multiply(loss_t2, w2)

    # Model joint loss
    #loss_op = tf.div(tf.add(l1, l2), 2)
    loss_op = tf.add(l1, l2)

    # L0 for task 1 and 2
    # workaround to assign value for first loss
    # Variables to hold vaue for initial task losses
    l01_ = tf.Variable(-1.0, trainable=False)
    l02_ = tf.Variable(-1.0, trainable=False)

    def assign_l01():
      with tf.control_dependencies([tf.assign(l01_, loss_t1)]):
        return tf.identity(l01_)
    def assign_l02():
      with tf.control_dependencies([tf.assign(l02_, loss_t2)]):
        return tf.identity(l02_)

    l01 = tf.cond(tf.equal(l01_, -1.0),
                  assign_l01,
                  lambda : tf.identity(l01_))
    l02 = tf.cond(tf.equal(l02_, -1.0),
                  assign_l02,
                  lambda : tf.identity(l02_))


    # L2-norm of gradients of each task loss wrt shared parameters
    G1R = tf.gradients(l1, weights_shared)
    G1 = tf.norm(G1R, ord=2)
    G2R = tf.gradients(l2, weights_shared)
    G2 = tf.norm(G2R, ord=2)

    # Gradient averaged over all tasks
    G_avg = tf.div(tf.add(G1, G2), 2)

    # Relative losses L_hat_i(t)
    l_hat_1 = tf.div(l1, l01)
    l_hat_2 = tf.div(l2, l02)
    l_hat_avg = tf.div(tf.add(l_hat_1, l_hat_2), 2)

    # Inverse training rates r_i(t)
    inv_rate_1 = tf.div(l_hat_1, l_hat_avg)
    inv_rate_2 = tf.div(l_hat_2, l_hat_avg)

    # Constant target (Eq. 1 in paper)
    a = tf.constant(alpha)
    C1 = tf.multiply(G_avg, tf.pow(inv_rate_1, a))
    C2 = tf.multiply(G_avg, tf.pow(inv_rate_2, a))
    C1 = tf.stop_gradient(tf.identity(C1))
    C2 = tf.stop_gradient(tf.identity(C2))

    # GradNorm loss (Eq. 2 in paper)
    loss_gradnorm = tf.add(
        tf.reduce_sum(tf.abs(tf.subtract(G1, C1))),
        tf.reduce_sum(tf.abs(tf.subtract(G2, C2))))

    # Renormalize weights
    with tf.control_dependencies([loss_gradnorm]):
        coef = tf.div(2.0, tf.add(w1, w2))
        w1_update = w1.assign(tf.multiply(w1, coef))
        w2_update = w2.assign(tf.multiply(w2, coef))
        update_op  = [w1_update, w2_update]


with tf.Session() as sess:
    # Initialize optimizers + variables and start train
    global_step = tf.train.get_or_create_global_step()

    # Define model optimizer
    model_vars = [weights_shared, bias_shared,
            weights_t1_1, bias_t1_1,
            weights_t1_2, bias_t1_2,
            weights_t1_3, bias_t1_3,
            weights_t2_1, bias_t2_1,
            weights_t2_2, bias_t2_2,
            weights_t2_3, bias_t2_3]

    opt_model = tf.train.AdamOptimizer(LR)
    train_step = opt_model.minimize(loss_op, global_step=global_step, var_list=model_vars)

    # Define loss optimizer
    loss_vars = [w1, w2]

    opt_loss = tf.train.AdamOptimizer(LR)
    loss_step = opt_loss.minimize(loss_gradnorm, global_step=global_step, var_list=loss_vars)

    sess.run(tf.global_variables_initializer())

    prob = 0.5

    weight1 = []
    weight2 = []

    for it in range(epoch):
        epoch_cost = 0.0
        epoch_cost1 = 0.0
        epoch_cost2 = 0.0

        num_minibatches = int(input_size / mb_size)
        minibatches = random_mini_batches(X_train, Y1_train, Y2_train, mb_size, seed=seed)

        for minibatch in minibatches:
            XE, YE1, YE2  = minibatch

            # Update GradNorm and model weights
            l, l_t1, l_t2, we1, we2, _ = \
                    sess.run([loss_op, l1, l2, w1, w2, [train_step, loss_step, update_op]],
                              feed_dict={X_: XE,
                                         Y1_: np.reshape(YE1, [-1, 1]),
                                         Y2_: np.reshape(YE2, [-1, 1]),
                                         keep_prob: 0.5})

            epoch_cost += (l / num_minibatches)
            epoch_cost1 += (l_t1 / num_minibatches)
            epoch_cost2 += (l_t2 / num_minibatches)

            weight1.append(we1)
            weight2.append(we2)

        costtr.append(epoch_cost)
        cost1tr.append(epoch_cost1)
        cost2tr.append(epoch_cost2)

        # Evaluate
        l_val, l1_val, l2_val, weights = \
                sess.run([loss_op, l1, l2, [w1, w2]],
                          feed_dict={X_: X_valid,
                                     Y1_: np.reshape(Y1_valid, [-1, 1]),
                                     Y2_: np.reshape(Y2_valid, [-1, 1])})
        cost_val.append(l_val)
        cost1_val.append(l1_val)
        cost2_val.append(l2_val)

        print('epoch:', it, 'loss:', costtr[-1])
        #print('Weight values:', weights)

    plt.plot(np.squeeze(costtr), '-r', np.squeeze(cost_val), '-b')
    plt.ylabel('total cost')
    plt.xlabel('iterations')
    plt.gca().legend(('train','dev'))
    plt.show()

    plt.plot(np.squeeze(cost1tr), '-r', np.squeeze(cost1_val), '-b')
    plt.ylabel('task 1 cost')
    plt.xlabel('iterations')
    plt.gca().legend(('train','dev'))
    plt.show()

    plt.plot(np.squeeze(cost2tr), '-r', np.squeeze(cost2_val), '-b')
    plt.ylabel('task 2 cost')
    plt.xlabel('iterations')
    plt.gca().legend(('train','dev'))
    plt.show()

    plt.plot(np.squeeze(weight1), '-r', np.squeeze(weight2), '-b')
    plt.ylabel('Weight Magnitude')
    plt.xlabel('steps')
    plt.gca().legend(('w1','w2'))
    plt.show()

