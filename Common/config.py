num_epochs = 5
num_workers = 10

idx_max_length = 50000
grad_shift = 2 ** 20

f = 1

topk = 40

gradient_frac = 2 ** 10
gradient_rand = 2 ** 8

server1_address = "127.0.0.1"
port1 = 50001


server2_address = "127.0.0.1"
port2 = 50002


mpc_idx_port = 50003
mpc_grad_port = 50004

grpc_options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]

# Ditto parameters
coef = 0.05 # local-global model lambda
llr = 0.01 # local learning rate

# adaptive clipping parameters
initClippingBound = 1 # initial clipping bound
blr = 0.2 # clipping bound learning rate
gamma = 0.5 # non-clipping ratio {0.1, 0.3, 0.5, 0.7, 0.9}
# b_noise = num_workers/20 # noise standard deviation added to counts (from server side)
b_noise = 1.5 # (from client side) [1.7823 ~ (e,10^-5)DP]
z_multiplier = 2 # {0, 0.01, 0.03, 0.1}

