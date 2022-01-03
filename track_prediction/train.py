import torch
import torch.nn as nn
import math,statistics,csv
import os
from config.config import parser
from dataloader import trainloader,trainset, testloader, testset
from VanillaRNN import VanillaRNN
from BayesianRNN import BayesianRNN

args = parser.parse_args()
input_dim = 1
hidden_dim = args.hidden_dim
output_dim = 1

NUM_BATCHES = len(trainloader)
NUM_TEST_BATCHES = len(testloader)
TRAIN_SIZE = len(trainset)
TEST_SIZE = len(testset)
epochs = args.epochs
results_file_path = os.path.join(os.getcwd(), 'results', args.result_file + '.csv')

if args.model == 'BRNN':
    model = BayesianRNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(1,epochs+1):
        single_loss=0
        MSE_loss = 0

        model.train()
        k=0
        for seq,labels,tracks in trainloader:
            batch_losses = []
            MSE_losses = []
            batch_losses_test = []
            MSE_losses_test = []
            model.zero_grad()
            loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss    
            MSE_losses.append(MSE_loss*len(seq))
            batch_losses.append(loss*len(seq))
        if len(MSE_losses)>0:
            MSE_losses = torch.stack(MSE_losses)
            batch_losses = torch.stack(batch_losses)
            print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')
    


    train_mse, test_mse = [], []
    loss_fn = torch.nn.MSELoss()
    SAMPLES = args.samples
    for seq, labels,tracks in trainloader:
        out = torch.zeros(SAMPLES, len(seq), output_dim, 2)
        for i in range(SAMPLES):
            out[i] = model(seq, sampling=True)
        out = out.mean(0)
        train_mse.append(torch.pow(out - labels, 2))


    train_rmse = torch.sqrt(torch.mean(torch.cat(train_mse, dim=0)))
    train_rmse_std = torch.std(torch.sqrt(torch.cat(train_mse, dim=0)))  


    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)
        for seq, labels,tracks in testloader:
            out = torch.zeros(SAMPLES, len(seq), output_dim, 2)
            for i in range(SAMPLES):
                out[i] = model(seq, sampling=True)
            p5_, p95_ =out.quantile(0.05, 0).tolist(), out.quantile(0.95, 0).tolist()
            out = out.mean(0) 
        

            for i,x,y,z,a,b in zip(tracks,seq, labels, out.tolist(), p5_, p95_):
                record = [i.item(), x[0].tolist()[0], x[0].tolist()[1], x[1].tolist()[0], x[1].tolist()[1], 
                        x[2].tolist()[0], x[2].tolist()[1], x[3].tolist()[0], x[3].tolist()[1], 
                        y.tolist()[0][0], y.tolist()[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
            test_mse.append(torch.pow(out - labels, 2))


    test_rmse = torch.sqrt(torch.mean(torch.cat(test_mse, dim=0)))
    test_rmse_std = torch.std(torch.sqrt(torch.cat(test_mse, dim=0)))    




    print('---------------------------')
    print('---------------------------')
    print(f'Train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'Train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')
    
else:
    model = VanillaRNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    Loss = nn.MSELoss()
    
    for i in range(1,epochs+1):
        single_loss=0
        MSE_loss = []

        model.train()
        for seq,labels, tracks in trainloader:
            model.zero_grad()
            Outputs = model(seq)
            loss = Loss(Outputs, labels)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss 
            MSE_loss.append(single_loss*len(seq))
        MSE_loss = torch.stack(MSE_loss)    
        print(f'epoch: {i:3} loss: {torch.sqrt(MSE_loss.sum()/TRAIN_SIZE):10.8f}') 

    
    train_mse, test_mse = [], []
    loss_fn = torch.nn.MSELoss()

    for seq, labels, tracks in trainloader:
        out = model(seq)
            
        train_mse.append(torch.pow(out - labels, 2))

    train_rmse = torch.sqrt(torch.mean(torch.cat(train_mse, dim=0)))
    train_rmse_std = torch.std(torch.sqrt(torch.cat(train_mse, dim=0)))  

        
    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)
        for seq, labels, tracks in testloader:
                    
            out= model(seq)
                    
            for i,x,y,z,a,b in zip(tracks, seq, labels, out.tolist(), out.tolist(), out.tolist()):
                        
                record = [i.item(), x[0].tolist()[0], x[0].tolist()[1], x[1].tolist()[0], x[1].tolist()[1], 
                        x[2].tolist()[0], x[2].tolist()[1], x[3].tolist()[0], x[3].tolist()[1], 
                        y.tolist()[0][0], y.tolist()[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
            test_mse.append(torch.pow(out - labels, 2))

    test_rmse = torch.sqrt(torch.mean(torch.cat(test_mse, dim=0)))
    test_rmse_std = torch.std(torch.sqrt(torch.cat(test_mse, dim=0)))    




    print('---------------------------')
    print('---------------------------')
    print(f'Train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'Train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')
    

