num_epochs = 51
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
# global learning rate
glr = 0.01
# local epochs
local_epoch = 3
# local learning rate
llr = 0.01 
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
b_noise_std = 5
grad_noise_sigma = 1.005

# HDBSCAN on/off
naive_aggregation = False

# data distribution
_noniid = False # q=0.3 by default
# differential privacy parameters
_dpoff = True
_dpcompen = False
delta = 1e-4
account_method = "autodp"

# inference attack
dp_test = False
dp_in = False
dp_client = '0'

# byzantine clients
flipping_clients = []
malicious_client = []
reconstruct_inference = False # replace the last client by default
target = 3

# deepsight
# the number of parameters of an output layer neuron to neurons of the previous layer
weight_index = 850
bias_index = 10

# recording description 
recording = False
surffix = ""