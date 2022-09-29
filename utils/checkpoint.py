#%% LIBRARIES

import os
import torch

#%% SAVE MODEL STATE

def save(model, logger, hps, k):
    # Create the path the checkpint will be saved at using the epoch number
    path = os.path.join(hps["model_save_dir"], str(k) + "-fold" + ".pt")

    # Create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        "logs": logger.get_logs(),
        "params": model.state_dict()
    }

    # Save checkpoint
    torch.save(checkpoint, path)
    
#%% RESTORE MODEL STATE
    
def restore(net, logger, hps, k):
    """ 
    Load back the model and logger from a given k
    fold detailed if hps["restore_model"] is available
    """

    path = os.path.join(hps["model_save_dir"], str(k) + "-fold" + ".pt")

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint["logs"])
            net.load_state_dict(checkpoint["params"])

            print("Network Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            
            hps["start_epoch"] = 0
    else:
        print("Restore point unavailable. Training from scratch.")
        
        hps["start_epoch"] = 0