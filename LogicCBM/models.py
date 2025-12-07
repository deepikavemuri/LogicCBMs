
from LogicCBM.template_model import MLP, inception_v3, End2EndModel, End2EndLogicModel
from difflogic import LogicLayer
import torch
from torch import nn

PROJ_PATH = '<PROJECT_PATH>'
# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                        three_class=three_class)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid, make_boolean, continue_training, continue_model=None):
    model1, model2 = None, None
    if continue_training:
        model = torch.load(continue_model)
        model1 = model.module.first_model
        model2 = model.module.sec_model
    else:
        model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                            n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                            three_class=(n_class_attr == 3))
        
        if n_class_attr == 3:
            model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
        else:
            model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr, make_boolean)

# Logic-based Model 
def ModelXtobCtoLtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid, n_logic_neurons, n_logic_layers, use_pretrained_logic, freeze_logic, 
                 use_pretrained_bb_con, dataset, use_aux_clf, ll_connections, discrete_gates, make_boolean, 
                 continue_training, concept_pairs):

    model1 = ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes,
                         expand_dim, use_relu, use_sigmoid, make_boolean, continue_training)
    if use_pretrained_bb_con:
        if dataset=="cub":
            trained_model = torch.load(f'{PROJ_PATH}/saved_models/cub_Joint_model.pth')
        elif dataset=="cifar100":
            trained_model = torch.load(f'{PROJ_PATH}/saved_models/cifar100_Joint_model.pth')
        elif dataset=="awa2":
            trained_model = torch.load(f'{PROJ_PATH}/saved_models/awa2_Joint_model.pth')
        model1 = trained_model
        
        if freeze:
            for param in model1.parameters():
                param.requires_grad = False 
    
    model2 = [LogicLayer(n_attributes, n_logic_neurons, discrete_gates=discrete_gates, concept_pairs=concept_pairs)]
    if ll_connections == "full" or ll_connections == "correlated":
        model2 = [LogicLayer(n_attributes, n_logic_neurons, connections=ll_connections, discrete_gates=discrete_gates, concept_pairs=concept_pairs)]
        for _ in range(n_logic_layers-1):
            model2.append(LogicLayer(n_logic_neurons, n_logic_neurons, connections=ll_connections, discrete_gates=discrete_gates, concept_pairs=concept_pairs))
        model2.append(nn.Linear(n_logic_neurons, num_classes))
    else:
        for _ in range(n_logic_layers-1):
            model2.append(LogicLayer(n_logic_neurons, n_logic_neurons, discrete_gates=discrete_gates, concept_pairs=concept_pairs)) 
        model2.append(nn.Linear(n_logic_neurons, num_classes))    
    model2 = nn.Sequential(*model2)

    if use_pretrained_logic:
        MODEL_PATH = ''
        state_dicts = torch.load(f'{PROJ_PATH}/saved_models/{MODEL_PATH}')

        model2.load_state_dict(state_dicts['logic_classifier'])
        if freeze_logic:
            for param in model2.parameters():
                param.requires_grad = False 

    return End2EndLogicModel(model1, model2, use_relu, use_sigmoid, n_class_attr, use_aux_clf, make_boolean)

# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux)

# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, use_aux, n_attributes, three_class, connect_CY):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=False, three_class=three_class,
                        connect_CY=connect_CY)
