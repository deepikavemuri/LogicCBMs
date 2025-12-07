"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import os
import sys
import argparse
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

import math
import torch
from LogicCBMs.analysis import Logger, AverageMeter, accuracy, binary_accuracy
import torchvision.transforms as transforms

from LogicCBM.dataset import find_class_imbalance, load_data, CUBDataset, Cifar100Loader, AnimalDataset
from LogicCBM.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from LogicCBM.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY, ModelXtobCtoLtoY

def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter1, acc_meter2, criterion, attr_criterion, args, is_training, acc_meter_aux=None):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        print("Evaluating")
        model.eval()

    for _, data in enumerate(tqdm(loader)):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels, dim=1).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)

            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var) 
                losses.append(loss_main)
                out_start = 1
                if args.use_aux_clf:
                    loss_main2 = 0.5 * (1.0 * criterion(outputs[1], labels_var) + 0.4 * criterion(aux_outputs[1], labels_var))
                    losses.append(loss_main2)
                    out_start = 2
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]) 
                                                           + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i])))
    
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
                if args.use_aux_clf:
                    loss_main2 = criterion(outputs[1], labels_var)
                    losses.append(loss_main2)
                    out_start = 2
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter1.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter1.update(acc[0], inputs.size(0))
            if args.use_aux_clf:
                acc = accuracy(outputs[1], labels, topk=(1,)) # bb-c-cls classifier
                acc_meter_aux.update(acc[0], inputs.size(0))

            if args.use_aux_clf:
                sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs[2:], dim=1))
            else:
                sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs[1:], dim=1))
            con_acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter2.update(con_acc.data.cpu().numpy(), inputs.size(0))
            
        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    if args.use_aux_clf:
        return loss_meter, acc_meter1, acc_meter2, acc_meter_aux
    return loss_meter, acc_meter1, acc_meter2

def train(model, args):
    # Determine imbalance
    imbalance = None 
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)
            
    model = model.cuda()
    run_name = "<RUN_NAME>"
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    if args.ckpt: #retraining
        train_loader = load_data(args.dataset, [train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, split='train', n_class_attr=args.n_class_attr, image_dir=args.image_dir, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data(args.dataset, [train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, split='train', image_dir=args.image_dir, \
                                n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data(args.dataset, [val_data_path], args.use_attr, args.no_img, args.batch_size, split='test', image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_con_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        if args.use_aux_clf:
            train_acc_meter2 = AverageMeter()
            train_loss_meter, train_acc_meter, train_con_acc_meter, train_acc_meter2 = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, train_con_acc_meter, criterion, attr_criterion, args, is_training=True, acc_meter_aux=train_acc_meter2)
        else:
            train_loss_meter, train_acc_meter, train_con_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, train_con_acc_meter, criterion, attr_criterion, args, is_training=True)

        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            val_con_acc_meter = AverageMeter()
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                if args.use_aux_clf:
                    val_acc_meter2 = AverageMeter()
                    val_loss_meter, val_acc_meter, val_con_acc_meter, val_acc_meter2 = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, val_con_acc_meter, criterion, attr_criterion, args, is_training=False, acc_meter_aux=val_acc_meter2)
                else:
                    val_loss_meter, val_acc_meter, val_con_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, val_con_acc_meter, criterion, attr_criterion, args, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())
        
        if (epoch % args.save_step == 0 and epoch > 0) or epoch == args.epochs-1:
            save_path = os.path.join(args.save_dir, run_name + '_%d_model.pth' % epoch)
            torch.save(model, save_path)

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break
    return {"val_acc": val_acc_meter.avg.detach().cpu().numpy()[0]}


def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    n_classes = 200
    if args.dataset == "cifar100":
        n_classes = 100
    if args.dataset == "awa2":
        n_classes = 50

    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=n_classes, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid, 
                         make_boolean=args.make_boolean, continue_training=args.continue_training, 
                         continue_model=args.continue_model)
    train(model, args)
    
def train_X_to_bC_to_Ly(args):
    n_classes = 200
    args.concept_pairs = None
    if args.dataset == "cub":
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
        dataset = CUBDataset([train_data_path, val_data_path], True, False, False, args.image_dir, args.n_class_attr, 'train', None)
        if args.ll_connections == "correlated":
            args.concept_pairs = CUBDataset.select_concept_pairs(dataset, num_pairs=args.n_logic_neurons)

    if args.dataset == "cifar100" or args.dataset == "imagenet100" or args.dataset == "imagenet50":
        n_classes = 100
        if args.ll_connections == "correlated":
            dataset = Cifar100Loader(split='train')
            args.concept_pairs = Cifar100Loader.select_concept_pairs(dataset, num_pairs=args.n_logic_neurons)

    if args.dataset == "awa2":
        n_classes = 50
        if args.ll_connections == "correlated":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            dataset = AnimalDataset(data_dir='/data1/ai22resch11001/projects/data/AWA2/Animals_with_Attributes2/', classes_file=None, transform=transform, split='train')
            args.concept_pairs = AnimalDataset.select_concept_pairs(dataset, num_pairs=args.n_logic_neurons)

    model = ModelXtobCtoLtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=n_classes, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid, 
                         n_logic_neurons=args.n_logic_neurons, n_logic_layers=args.n_logic_layers, use_pretrained_logic=args.use_pretrained_logic,
                         freeze_logic = args.freeze_logic, use_pretrained_bb_con = args.use_pretrained_bb_con, dataset=args.dataset, use_aux_clf=args.use_aux_clf,
                         ll_connections=args.ll_connections, discrete_gates=args.discrete_gates, make_boolean=args.make_boolean, continue_training=args.continue_training, 
                         concept_pairs=args.concept_pairs)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)


def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='LogicCBM Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe', 'Logic',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')


    parser.add_argument('-save_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, default=32, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
    parser.add_argument('-freeze_logic', action='store_true', help='whether to freeze the logic part')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-discrete_gates', action='store_true',
                        help='whether to use discrete or soft logic gates during training')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-uncertain_labels', action='store_true',
                        help='whether to use (normalized) attribute certainties as labels')
    parser.add_argument('-use_pretrained_logic', action='store_true', help='use a pretrained logic to class mapping and finetune')
    parser.add_argument('-use_pretrained_bb_con', action='store_true', help='use a pretrained backbone and concept layer')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-n_logic_neurons', type=int, default=16000, help='number of neurons in the logic layer')
    parser.add_argument('-n_logic_layers', type=int, default=1, help='number of logic gate layers to add')    
    parser.add_argument('-expand_dim', type=int, default=0,
                        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='/data1/ai22resch11001/projects/data/CUB_200_2011/CUB_processed', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ll_connections', default='random', help='Type of connections initialized to the logic layer: random or correlated')
    parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-make_boolean', action='store_true', help='Make it a boolean CBM')
    parser.add_argument('-continue_training', action='store_true', help='Continue training an already trained joint model')
    parser.add_argument('-continue_model', type=str, default=None, help='The model path to continue training on')
    parser.add_argument('-use_aux_clf', action='store_true',help='indicates whether the logic neurons have weighted or not')
    parser.add_argument('-connect_CY', action='store_true',
                        help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    return (args,)