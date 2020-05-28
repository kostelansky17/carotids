import torch

def accuracy(output, label):
    _, predicted = torch.max(output.data, 1)

    return (predicted == label).sum().item() / len(label)


def iou(labels, outputs, treshold):
    size = len(labels)

    correct = 0
    for i in range(size):
      if outputs[i, 0] >= outputs[i, 2] or \
         outputs[i, 1] >= outputs[i, 3]:
        continue
      
      x_0 = max(labels[i, 0], outputs[i, 0])
      y_0 = max(labels[i, 1], outputs[i, 1])
      
      x_1 = min(labels[i, 2], outputs[i, 2])
      y_1 = min(labels[i, 3], outputs[i, 3])


      if x_1 < x_0 or y_1 < y_0:
        continue

      intersection_area = (x_1 - x_0) * (y_1 - y_0)

      label_area = (labels[i, 2] - labels[i, 0]) * (labels[i, 3] - labels[i, 1])
      output_area = (outputs[i, 2] - outputs[i, 0]) * (outputs[i, 3] - outputs[i, 1])

      iou = intersection_area / float(label_area + output_area - intersection_area)

      if iou >= treshold:
        correct += 1

    return correct


def evaluate_dataset_iou(model, dataset, device):
    model.eval()
    s = 0 
    for inputs, label in dataset:
      
        with torch.no_grad():
            prediction = model([inputs.to(device)])
          
        prediction = prediction[0]["boxes"]
        inputs = label["boxes"]
        s += iou(inputs.int(), prediction.int(), .85)

    print(f"Validation: {s/len(dataset)}")