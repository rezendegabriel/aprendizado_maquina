#%% LIBRARIES

from models import mlp
from utils.checkpoint import restore
from utils.logger import Logger

#%% SETUP MODEL

models = {
    "mlp": mlp.MLP,
}

def setup_model(hps, k = None):
    #Hyper-parameters
    num_inputs = hps["num_inputs"]
    num_outputs = hps["num_outputs"]
    layer_size = hps["layer_size"]
    drop = hps["drop"]
    
    model = models[hps["model"]](num_inputs, num_outputs, layer_size, drop)
    logger = Logger()
    
    if hps["restore_model"] == "y":
        restore(model, logger, hps, k)
        
    return logger, model