import hydra
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.train_utils import load_train_df, load_test_df
from hyp3.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor
from hyp3.model import LogRegModel
from hyp3.embedding import Embedder

@hydra.main(version_base=None, config_path="../../config/hyp3", config_name='config_3')
def train(cfg):
    #Pandas DF
    train_df=load_train_df(cfg)
    test_df=load_test_df(cfg)
    
    #Features
    train_df, test_df = add_features(train_df, test_df)

    #Manually fill in missing values
    train_df = predict_missing_values(train_df)
    test_df = predict_missing_values(test_df)

    #Spending Features
    train_df, test_df = add_spending(train_df, test_df)

    #X and y
    X = train_df[features]
    y = train_df['Transported'].astype(int)

    #Create test DF
    X_test = test_df[features]

    #Split training data into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=cfg.train_parameters.test_size, random_state=cfg.train_parameters.state)

    #Normalizing
    preprocessor = get_preprocessor()
    all_data = pd.concat([X_train, X_test])
    preprocessor.fit(all_data)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)


    #Tensors
    X_train_features = torch.tensor(X_train, dtype = torch.float32)
    y_train_features = torch.tensor(y_train.values, dtype = torch.long)

    X_valid_features = torch.tensor(X_valid, dtype =torch.float32)
    y_valid_features = torch.tensor(y_valid.values, dtype = torch.long)
    X_test_features = torch.tensor(X_test, dtype = torch.float32)

    #Embedding
    # embedding_dim = 5
    # num_cols = 5
    # output_dim = 1
    # embed_model = Embedder(num_cols, embedding_dim, output_dim)
    # embed_criterion = nn.BCEWithLogitsLoss()
    # embed_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # embed_epochs = 100
    # for epoch in range (embed_epochs):
    #     embed_model.train()
    #     embed_optimizer.zero_grad()
    #     output = model(X_train_features)
    #     loss = 
    #     loss.backward()
    #     embed_optimizer.step()
    
    #NN
    batch_size = cfg.model.batch

    num_epochs=int(cfg.model.iters/ (len(X_train)/batch_size))
    train_dataset = torch.utils.data.TensorDataset(X_train_features, y_train_features)
    valid_dataset = torch.utils.data.TensorDataset(X_valid_features, y_valid_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    input_dim= X_train.shape[1]
    model = LogRegModel(input_dim)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

    count = 0
    loss_list = []
    iteration_list = []

    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate (train_loader):
            outputs = model(data)
            loss = error(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1

            if count%50 == 0:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, labels in valid_loader: 
                        outputs = model(data)
                        pred = torch.max(outputs, 1)[1]
                        total += len(labels)
                        correct += (pred == labels).sum().item()
                accuracy = 100 * correct /total
                loss_list.append(loss.data)
                iteration_list.append(count)
        
            if count % 100 == 0:
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(count, loss.data, accuracy))


    #Predict
    outputs =  model(X_test_features)
    prediction = torch.max(outputs, 1)[1].numpy()  

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': prediction.astype(bool)
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    train()

