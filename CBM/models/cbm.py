import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowGMMConceptExtractor(nn.Module):
    """
    Wraps the Normalizing Flow and GMM to map x -> c.
    """
    def __init__(self, flow_model, gmm_model):
        super().__init__()
        self.flow = flow_model
        self.gmm = gmm_model

    def forward(self, x):
        # 1. Pass input through the normalizing flow to get latent z
        z, sldj = self.flow(x)
        
        # 2. Flatten z for the GMM classification
        z_flat = z.view(z.shape[0], -1)
        
        # 3. Get concept predictions and logits from GMM
        preds, complete_logits = self.gmm.classify(z_flat)
        
        # 4. Convert logits to continuous probabilities (Softmax)
        # complete_logits is a list of tensors
        c_probs = [F.softmax(logits, dim=-1) for logits in complete_logits]
        
        # Concatenate: Shape (Batch, 10+10+7+7) = (Batch, 34)
        c_continuous = torch.cat(c_probs, dim=-1)

        # Return everything needed
        return z, sldj, c_continuous, preds

class TaskPredictor(nn.Module):
    """
    A simple MLP mapping concepts c -> task y.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, c):
        return self.net(c)

class ModularCBM(nn.Module):
    """
    The complete Concept Bottleneck Model.
    """
    def __init__(self, concept_extractor, task_predictor, cf_encoder=None, num_y_classes=None):
        super().__init__()
        self.concept_extractor = concept_extractor
        self.task_predictor = task_predictor
        # Optional: For Phase 2 (Counterfactuals)
        self.cf_encoder = cf_encoder 
        self.num_y_classes = num_y_classes

    def forward(self, x, true_c_onehot=None, true_y=None, generate_cf=False):
        """
        Forward pass supporting Independent training and Counterfactual generation.
        """
        # 1. Extract concepts (x -> c)
        z, sldj, c_continuous, preds = self.concept_extractor(x)
        
        # 2. Determine input for Task Predictor
        # If true_c_onehot is provided (Independent Training), use it.
        # Otherwise, use the predicted c_continuous.
        if true_c_onehot is not None:
            c_input = true_c_onehot
        else:
            c_input = c_continuous
            
        # 3. Predict Task (c -> y)
        y_pred = self.task_predictor(c_input)
        
        # 4. Counterfactual Logic (Placeholder for Phase 2)
        # This prevents the error by accepting the arguments, even if we don't use them yet.
        cf_outputs = None
        if generate_cf and self.cf_encoder is not None and true_y is not None:
            # We will implement the CF logic here later
            pass

        # Return 6 values to match your training script unpacking:
        # y_pred, _, _, _, _, _ = cbm(...)
        return y_pred, c_continuous, preds, z, sldj, cf_outputs



# only for 'standard' paradigm:

class FlowLinearConceptExtractor(nn.Module):
    def __init__(self, flow_model, z_dim, c_dim_list):
        super().__init__()
        self.flow = flow_model
        total_c_dim = sum(c_dim_list) # 34
        
        # Simple Linear Mapping: 4704 -> 34
        # We REMOVE the "splitting" logic and Softmax. 
        # We just want a dense representation.
        self.connector = nn.Linear(z_dim, total_c_dim)

    def forward(self, x):
        # x -> z
        z, sldj = self.flow(x)
        z_flat = z.view(z.shape[0], -1)
        
        # z -> c (Raw continuous features)
        # NO SOFTMAX. This allows gradients to flow freely.
        c_continuous = self.connector(z_flat)
        
        # We return zeros for 'preds' because "concept accuracy" 
        # is meaningless in the Standard paradigm (we aren't learning real concepts)
        preds = torch.zeros(x.size(0), 4, device=x.device)

        return z, sldj, c_continuous, preds