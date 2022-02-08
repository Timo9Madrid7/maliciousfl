num_epochs = 21
total_number_clients = 200
num_workers = 20

# Model Setting
Model = "LeNet"
if Model == "LeNet":
    local_models_path = "./Model/LeNet/Local_Models/LeNet_"
    global_models_path = "./Model/LeNet/LeNet"
elif Model == "ResNet":
    local_models_path = "./Model/ResNet/Local_models/ResNet_"
    global_models_path = "./Model/ResNet/ResNet"
# local epochs
local_epoch = 3
# local learning rate
llr = 0.1 
# adaptive Ditto parameters
minLambda = 0.00
maxLambda = 2.00
# global Ditto parameters
global_lambda = 0


# adaptive clipping parameters
initClippingBound = 10 # initial clipping bound
beta = 0.1 # last round gradient weight
blr = 0.5 # clipping bound learning rate
gamma = 0.5 # non-clipping ratio {0.1, 0.3, 0.5, 0.7, 0.9}
# b_noise = num_workers/20 # noise standard deviation added to counts (from server side)
b_noise_std = 5 # (from client side) [1.7823 ~ (e,10^-5)DP]
grad_noise_sigma = 1.005 # {0, 0.01, 0.03, 0.1} * num_workers

# differential privacy parameters
_noniid = True
_dpoff = True
_dpcompen = False
# TODO: differential privacy test has not been implemented in this version
# _dprecord, _dpin, _dpclient = False, False, "0" #_dpin: whether _dpclient is involved in the training
delta = 1e-4
account_method = "autodp"

# byzantine clients
flipping_clients = []
malicious_client = []

# deepsight
# the number of parameters of an output layer neuron to neurons of the previous layer
weight_index = 850
bias_index = 10

# recording description 
recording = False
surffix = "dpon_compen_malicious"