import torch
import torch.nn as nn
import math,statistics,csv
import os
from config.config import parser
from dataloader import trainloader,trainset, testloader, testset
from VanillaRNN import VanillaRNN
from BayesianRNN import BayesianRNN
from VanillaLSTM import VanillaLSTM
from BayesianLSTM import BayesianLSTM

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
        batch_losses = []
        MSE_losses = []
        model.train()
        k=0
        for seq,labels,tracks in trainloader:
            model.zero_grad()
            loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss    
            MSE_losses.append(MSE_loss*len(seq))
            batch_losses.append(loss*len(seq))
            
        if len(MSE_losses)>0:
            #print(MSE_losses)
            MSE_losses = torch.FloatTensor(MSE_losses)
            batch_losses = torch.FloatTensor(batch_losses)
            #print(MSE_losses)
            print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')

        batch_losses_test, MSE_losses_test =[], []
        single_loss_test = 0
        MSE_loss_test = 0
        
        SAMPLES = args.samples
    

    
    labels_list,out_list=[],[]
    for seq, labels, tracks in trainloader:
        out_ = model.testing(seq)
        labels_list.append(labels)
        out_list.append(torch.swapaxes(out_,0,1))
    
    labels_list=torch.cat(labels_list, dim=0)
    out_list=torch.cat(out_list, dim=0)
    out_list = torch.swapaxes(out_list,0,1) 
    print(out_list.size())
    train_rmse_losses=[] 
    for i in range(SAMPLES):
        train_rmse_losses.append(torch.sqrt(nn.functional.mse_loss(out_list[i], labels_list)))
    
    train_rmse_losses = torch.FloatTensor(train_rmse_losses)
    out_list = out_list.quantile(0.5, 0)
    

    train_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
    train_rmse_std = torch.std(train_rmse_losses)


    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)

        labels_list,out_list=[],[]
        all_tracks,sequences= [],[]
        for seq, labels, tracks in testloader:
            out_ = model.testing(seq)
            labels_list.append(labels)
            sequences.append(seq)
            all_tracks.append(tracks)
            out_list.append(torch.swapaxes(out_,0,1))
        
        labels_list=torch.cat(labels_list, dim=0)
        sequences=torch.cat(sequences, dim=0)
        all_tracks=torch.cat(all_tracks, dim=0)
        out_list=torch.cat(out_list, dim=0)
        out_list = torch.swapaxes(out_list,0,1) 
        print(out_list.size())
        output = out_list
        test_rmse_losses=[] 
        for i in range(SAMPLES):
            test_rmse_losses.append(torch.sqrt(nn.functional.mse_loss(out_list[i], labels_list)))
        
        test_rmse_losses = torch.FloatTensor(test_rmse_losses)
        out_list = out_list.quantile(0.5, 0)
        

        test_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
        all_tracks = all_tracks.tolist()
        sequences = sequences.tolist()
        test_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
        labels_list = labels_list.tolist()
        p5_, p95_,out =output.quantile(0.05, 0).tolist(), output.quantile(0.95, 0).tolist(), output.quantile(0.5, 0).tolist()
        

        for i,x,y,z,a,b in zip(all_tracks,sequences, labels_list, out, p5_, p95_):
                record = [i, x[0][0], x[0][1], x[1][0], x[1][1], 
                        x[2][0], x[2][1], x[3][0], x[3][1], 
                        y[0][0], y[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
        


    
    test_rmse_std = torch.std(test_rmse_losses)



    print('---------------------------')
    print('---------------------------')
    print(f'train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')  
        
    


    
elif args.model=='RNN':
    model = VanillaRNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    Loss = nn.MSELoss()


    for i in range(1,epochs+1):
        single_loss=0
        MSE_loss = 0
        batch_losses = []
        MSE_losses = []
        model.train()
        k=0
        for seq,labels,tracks in trainloader:
            model.zero_grad()
            Outputs = model(seq)
            loss = Loss(Outputs, labels)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss    
            MSE_losses.append(loss.item()*len(seq))
            
            
        if len(MSE_losses)>0:
            #print(MSE_losses)
            MSE_losses = torch.FloatTensor(MSE_losses)
            #print(MSE_losses)
            print(f'epoch: {i:3} loss:  RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')

        batch_losses_test, MSE_losses_test =[], []
        single_loss_test = 0
        MSE_loss_test = 0
        
    

        
    train_mse, test_mse = [], []
    loss_fn = torch.nn.MSELoss()
    SAMPLES = args.samples
    output = []
    labels_total = []
    
    
    output = []
    train_rmse_losses = []
    for i in range(SAMPLES):
        out = []
        train_mse = []
        for seq, labels, tracks in trainloader:
            out_  = model(seq)
            out.append(out_)
            train_mse.append(nn.functional.mse_loss(out_,labels)*len(seq))
            
        out = torch.cat(out, dim=0)
        output.append(out)
        train_mse = torch.FloatTensor(train_mse)
        rmse_tr = torch.sqrt(train_mse.sum()/TRAIN_SIZE)
        train_rmse_losses.append(rmse_tr)
       
    
    train_rmse_losses = torch.FloatTensor(train_rmse_losses)
    train_rmse = torch.mean(train_rmse_losses)
    train_rmse_std = torch.std(train_rmse_losses)


    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)

        output = []
        test_rmse_losses = []     
        sequences = []
        all_labels = []
        all_tracks = []
        for i in range(SAMPLES):
            out = []
            test_mse =[]     
            sequences_ = []
            all_tracks_ = []
            all_labels_ = []
            for seq,labels, tracks in testloader:
                out_  = model(seq)
                out.append(out_)
                test_mse.append(nn.functional.mse_loss(out_,labels)*len(seq))
                sequences_.append(seq)
                all_labels_.append(labels)
                all_tracks_.append(tracks)
                
            out = torch.cat(out, dim=0)
            sequences_ = torch.cat(sequences_, dim=0)
            all_tracks_ = torch.cat(all_tracks_, dim=0)
            all_labels_ = torch.cat(all_labels_, dim=0)
            output.append(out)
            sequences.append(sequences_)
            all_labels.append(all_labels_)
            all_tracks.append(all_tracks_)
            test_mse = torch.FloatTensor(test_mse)
            rmse_tr = torch.sqrt(test_mse.sum()/TEST_SIZE)
            test_rmse_losses.append(rmse_tr)
            
            
        test_rmse_losses = torch.FloatTensor(test_rmse_losses)
        test_rmse = torch.mean(test_rmse_losses)
        test_rmse_std = torch.std(test_rmse_losses)
        output = torch.stack(output)
        sequences = torch.stack(sequences)
        all_labels = torch.stack(all_labels)
        all_tracks = torch.stack(all_tracks)
        p5_, p95_,out =output.quantile(0.05, 0).tolist(), output.quantile(0.95, 0).tolist(), output.quantile(0.5, 0).tolist()
        

        for i,x,y,z,a,b in zip(all_tracks[0],sequences[0], all_labels[0], out, p5_, p95_):
                record = [i.item(), x[0].tolist()[0], x[0].tolist()[1], x[1].tolist()[0], x[1].tolist()[1], 
                        x[2].tolist()[0], x[2].tolist()[1], x[3].tolist()[0], x[3].tolist()[1], 
                        y.tolist()[0][0], y.tolist()[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
        


    test_rmse_losses = torch.FloatTensor(test_rmse_losses)
    test_rmse = torch.mean(test_rmse_losses)
    test_rmse_std = torch.std(test_rmse_losses)



    print('---------------------------')
    print('---------------------------')
    print(f'train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')

elif args.model == 'LSTM':
    model = VanillaLSTM(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    Loss = nn.MSELoss()


    for i in range(1,epochs+1):
        single_loss=0
        MSE_loss = 0
        batch_losses = []
        MSE_losses = []
        model.train()
        k=0
        for seq,labels,tracks in trainloader:
            model.zero_grad()
            Outputs = model(seq)
            loss = Loss(Outputs, labels)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss    
            MSE_losses.append(loss.item()*len(seq))
            
            
        if len(MSE_losses)>0:
            #print(MSE_losses)
            MSE_losses = torch.FloatTensor(MSE_losses)
            #print(MSE_losses)
            print(f'epoch: {i:3} loss:  RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')

        batch_losses_test, MSE_losses_test =[], []
        single_loss_test = 0
        MSE_loss_test = 0
        
    

        
    train_mse, test_mse = [], []
    loss_fn = torch.nn.MSELoss()
    SAMPLES = args.samples
    output = []
    labels_total = []
    
    
    output = []
    train_rmse_losses = []
    for i in range(SAMPLES):
        out = []
        train_mse = []
        for seq, labels, tracks in trainloader:
            out_  = model(seq)
            out.append(out_)
            train_mse.append(nn.functional.mse_loss(out_,labels)*len(seq))
            
        out = torch.cat(out, dim=0)
        output.append(out)
        train_mse = torch.FloatTensor(train_mse)
        rmse_tr = torch.sqrt(train_mse.sum()/TRAIN_SIZE)
        train_rmse_losses.append(rmse_tr)
       
    
    train_rmse_losses = torch.FloatTensor(train_rmse_losses)
    train_rmse = torch.mean(train_rmse_losses)
    train_rmse_std = torch.std(train_rmse_losses)


    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)

        output = []
        test_rmse_losses = []     
        sequences = []
        all_labels = []
        all_tracks = []
        for i in range(SAMPLES):
            out = []
            test_mse =[]     
            sequences_ = []
            all_tracks_ = []
            all_labels_ = []
            for seq,labels, tracks in testloader:
                out_  = model(seq)
                out.append(out_)
                test_mse.append(nn.functional.mse_loss(out_,labels)*len(seq))
                sequences_.append(seq)
                all_labels_.append(labels)
                all_tracks_.append(tracks)
                
            out = torch.cat(out, dim=0)
            sequences_ = torch.cat(sequences_, dim=0)
            all_tracks_ = torch.cat(all_tracks_, dim=0)
            all_labels_ = torch.cat(all_labels_, dim=0)
            output.append(out)
            sequences.append(sequences_)
            all_labels.append(all_labels_)
            all_tracks.append(all_tracks_)
            test_mse = torch.FloatTensor(test_mse)
            rmse_tr = torch.sqrt(test_mse.sum()/TEST_SIZE)
            test_rmse_losses.append(rmse_tr)
            
            
        test_rmse_losses = torch.FloatTensor(test_rmse_losses)
        test_rmse = torch.mean(test_rmse_losses)
        test_rmse_std = torch.std(test_rmse_losses)
        output = torch.stack(output)
        sequences = torch.stack(sequences)
        all_labels = torch.stack(all_labels)
        all_tracks = torch.stack(all_tracks)
        p5_, p95_,out =output.quantile(0.05, 0).tolist(), output.quantile(0.95, 0).tolist(), output.quantile(0.5, 0).tolist()
        

        for i,x,y,z,a,b in zip(all_tracks[0],sequences[0], all_labels[0], out, p5_, p95_):
                record = [i.item(), x[0].tolist()[0], x[0].tolist()[1], x[1].tolist()[0], x[1].tolist()[1], 
                        x[2].tolist()[0], x[2].tolist()[1], x[3].tolist()[0], x[3].tolist()[1], 
                        y.tolist()[0][0], y.tolist()[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
        


    test_rmse_losses = torch.FloatTensor(test_rmse_losses)
    test_rmse = torch.mean(test_rmse_losses)
    test_rmse_std = torch.std(test_rmse_losses)



    print('---------------------------')
    print('---------------------------')
    print(f'train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')

else:
    model = BayesianLSTM(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(1,epochs+1):
        single_loss=0
        MSE_loss = 0
        batch_losses = []
        MSE_losses = []
        model.train()
        k=0
        for seq,labels,tracks in trainloader:
            model.zero_grad()
            loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
            loss.backward(retain_graph = True)
            optimizer.step()
            single_loss = loss    
            MSE_losses.append(MSE_loss*len(seq))
            batch_losses.append(loss*len(seq))
            
        if len(MSE_losses)>0:
            #print(MSE_losses)
            MSE_losses = torch.FloatTensor(MSE_losses)
            batch_losses = torch.FloatTensor(batch_losses)
            #print(MSE_losses)
            print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')

        batch_losses_test, MSE_losses_test =[], []
        single_loss_test = 0
        MSE_loss_test = 0
        
    

        
    SAMPLES = args.samples
    

    
    labels_list,out_list=[],[]
    for seq, labels, tracks in trainloader:
        out_ = model.testing(seq)
        labels_list.append(labels)
        out_list.append(torch.swapaxes(out_,0,1))
    
    labels_list=torch.cat(labels_list, dim=0)
    out_list=torch.cat(out_list, dim=0)
    out_list = torch.swapaxes(out_list,0,1) 
    print(out_list.size())
    train_rmse_losses=[] 
    for i in range(SAMPLES):
        train_rmse_losses.append(torch.sqrt(nn.functional.mse_loss(out_list[i], labels_list)))
    
    train_rmse_losses = torch.FloatTensor(train_rmse_losses)
    out_list = out_list.quantile(0.5, 0)
    

    train_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
    train_rmse_std = torch.std(train_rmse_losses)


    with open(results_file_path, 'w') as f:
        writer = csv.writer(f)
        header = ['track_id', 'input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
                'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
                'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
                '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude']
        writer.writerow(header)

        labels_list,out_list=[],[]
        all_tracks,sequences= [],[]
        for seq, labels, tracks in testloader:
            out_ = model.testing(seq)
            labels_list.append(labels)
            sequences.append(seq)
            all_tracks.append(tracks)
            out_list.append(torch.swapaxes(out_,0,1))
        
        labels_list=torch.cat(labels_list, dim=0)
        sequences=torch.cat(sequences, dim=0)
        all_tracks=torch.cat(all_tracks, dim=0)
        out_list=torch.cat(out_list, dim=0)
        out_list = torch.swapaxes(out_list,0,1) 
        print(out_list.size())
        output = out_list
        test_rmse_losses=[] 
        for i in range(SAMPLES):
            test_rmse_losses.append(torch.sqrt(nn.functional.mse_loss(out_list[i], labels_list)))
        
        test_rmse_losses = torch.FloatTensor(test_rmse_losses)
        out_list = out_list.quantile(0.5, 0)
        

        test_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
        all_tracks = all_tracks.tolist()
        sequences = sequences.tolist()
        test_rmse = torch.sqrt(nn.functional.mse_loss(out_list,labels_list))
        labels_list = labels_list.tolist()
        p5_, p95_,out =output.quantile(0.05, 0).tolist(), output.quantile(0.95, 0).tolist(), output.quantile(0.5, 0).tolist()
        

        for i,x,y,z,a,b in zip(all_tracks,sequences, labels_list, out, p5_, p95_):
                record = [i, x[0][0], x[0][1], x[1][0], x[1][1], 
                        x[2][0], x[2][1], x[3][0], x[3][1], 
                        y[0][0], y[0][1], z[0][0], z[0][1], 
                        a[0][0], a[0][1], b[0][0], b[0][1]]
                writer.writerow(record)
        


    
    test_rmse_std = torch.std(test_rmse_losses)



    print('---------------------------')
    print('---------------------------')
    print(f'train RMSE ----> {train_rmse}')
    print(f'Test RMSE ----> {test_rmse}')
    print(f'train RMSE std ----> {train_rmse_std}')
    print(f'Test RMSE std----> {test_rmse_std}')  
















































   
 