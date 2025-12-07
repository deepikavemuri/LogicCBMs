import torch 
import sys
sys.path.append('<PROJECT_PATH>')

def gate_to_str(a, b, i):
    if i == 0:
        return 'F'
    elif i == 1:
        return f'{a} AND {b}'
    elif i == 2:
        return f'NOT({a} IMPLIES {b})'
    elif i == 3:
        return f'{a}'
    elif i == 4:
        return f'NOT({b} IMPLIES {a})'
    elif i == 5:
        return f'{b}'
    elif i == 6:
        return f'{a} XOR {b}'
    elif i == 7:
        return f'{a} OR {b}'
    elif i == 8:
        return f'NOT({a} OR {b})'
    elif i == 9:
        return f'NOT({a} XOR {b})'
    elif i == 10:
        return f'NOT({b})'
    elif i == 11:
        return f'{b} IMPLIES {a}'
    elif i == 12:
        return f'NOT({a})'
    elif i == 13:
        return f'{a} IMPLIES {b}'
    elif i == 14:
        return f'NOT({a} AND {b})'
    elif i == 15:
        return 'T'

def get_concepts(dataset):
    concepts = []
    if dataset == "cub":
        with open('./data/CUB_200_2011/attributes.txt', 'r') as f:
            for line in f:
                concepts.append(line.split()[1])
    elif dataset == "awa2":
        with open('./data/AWA2/Animals_with_Attributes2/predicates.txt', 'r') as f:
            for line in f:
                concepts.append(line.split()[1])
    return concepts

def get_class(dataset, cls_id):
    classes = []
    if dataset == "cub":
        with open('./data/CUB_200_2011/classes.txt', 'r') as f:
            for line in f:
                classes.append(line.split()[1].split('.')[1])
    elif dataset == "awa2":
        with open('./data/AWA2/Animals_with_Attributes2/testclasses.txt', 'r') as f:
            for line in f:
                classes.append(line.strip())
    return classes[cls_id]

def get_logic(model_path, cls_id):
    model = torch.load(model_path)
    logic_weights = model.sec_model[1].weight[cls_id]
    # Get the list of learnt logic gates
    gates = torch.argmax(model.sec_model[0].weights, dim=1)
    top2_weights, top2_indices = torch.topk(model.sec_model[0].logic_weights.weight, 2, dim=1)
    
    class_name = get_class("cub", cls_id)
    concepts = get_concepts("cub")
    predicates = []

    for i in range(250):
        predicates.append(gate_to_str(
            concepts[top2_indices[i][0].item()],
            concepts[top2_indices[i][1].item()],
            gates[i]
        ))

    print("Class: ", class_name)

    weighttopred = dict(zip(logic_weights.tolist(), predicates))
    weighttopred = sorted(weighttopred.items())

    top10, bottom10 = weighttopred[:10], weighttopred[::-1][:10]
    for i in range(10):
        print(top10[i])
    for i in range(10):
        print(bottom10[i])

if __name__ == "__main__":
    cls_id = 5
    model_path = '<MODEL_PATH>'
    get_logic(model_path, cls_id)