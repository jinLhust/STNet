import os
import torch
from torch import optim
from torch.utils.data import DataLoader

from module.Netstruc import STNet, TemNet
from utils.DataSet import Samples

if __name__ == '__main__':
    num_class = 2
    auto = False
    batch_size = 64 if not auto else 32
    EPOCH = 90
    save_weights = r'output/train_network/SpTmConv/1(1656, 64, 64, 64, 8, 72, 5, 0.1, 2, 5, 0.0, False, ' \
                   r'0.1)/model(90, 82.1, 6.152, 38.255).pth'  # your model path
    assert os.path.exists(save_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STNet(d_model=1656, d_bq=64, d_q=64, d_v=64, n_heads=8,
                          d_ff=72, n_layers=5, n_class=num_class, dropout=0.1,
                          kernel=5, ae_dropout=0.0, auto=auto, mse_thres=0.1)
    # model = TemNet(d_model=1656, d_bq=64, d_q=64, d_v=64, n_heads=8,
    #                  d_ff=72, n_layers=5, n_class=num_class, dropout=0.1,
    #                  kernel=5, ae_dropout=0.0, auto=auto, mse_thres=0.1)

    model.load_state_dict(torch.load(save_weights, map_location=device))
    model.to(device)
    model.eval()

    test_path = r'data/scatter/test'
    test_dataset = Samples(data_dir=[test_path])

    testLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print('model name: %s' % model.name)
    print('parameters: %s' % model.para_str)

    mse = 0
    max_mse = 0
    extra_cls_mse = 0
    extra_cls_min_mse = 1e5
    correct = 0
    total = 0
    total_extra = 0
    class_correct = torch.zeros(num_class).to(device)
    class_total = torch.zeros(num_class).to(device)
    internal_diff = []
    extra_diff = []
    internal_conf = []
    extra_conf = []
    internal_2nd_conf = []
    extra_2nd_conf = []
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels - 1
            labels[labels[:] <= 0] = 0
            labels = labels.to(device)
            outputs = model(inputs)
            for i in range(labels.size(0)):
                if labels[i] < (num_class - 1):
                    total += 1
                    internal_conf.append(model.y_ori[i, labels[i]])
                    internal_2nd_conf.append(model.y_ori[i, model.y_ori[i, :] < model.y_ori[i, :].max()].max())
                else:
                    total_extra += 1
                    extra_conf.append(model.y_ori[i, :].max())
                    extra_2nd_conf.append(model.y_ori[i, model.y_ori[i, :] < model.y_ori[i, :].max()].max())
            _, predicted = torch.max(outputs.data, dim=-1)
            comp = predicted.eq(labels)
            correct += comp.sum().item()
            for i in range(num_class):
                idx = labels.eq(i)
                class_total[i] += idx.sum()
                class_correct[i] += comp[idx].sum()
        mse /= total
        acc = correct / (total + total_extra) * 100
        class_acc = class_correct / class_total.clamp_min(1e-4) * 100
    print('Test performance: \tacc=%.2f \tmaxmse=%.6f \tmax_maxmse=%.6f' % (acc, mse, max_mse))
    print('\t\tClass accuracy: \t[' + ', '.join('%.2f' % class_acc[i].item() for i in range(num_class)) + ']')
