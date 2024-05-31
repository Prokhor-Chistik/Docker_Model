import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier


# Model Creating Function
def model_creating():

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    print('Train data imported...')

    X, y = train.drop(['binary_target', 'client_id', 'mrg_'], axis=1), train['binary_target']
    X[['регион', 'использование', 'pack']] = X[['регион', 'использование', 'pack']].fillna('Nan')
    
    model = CatBoostClassifier(iterations=3, learning_rate=0.3, depth=3, l2_leaf_reg = 3.928965, class_weights = {0: 1, 1: 2}, cat_features=["регион", "использование", "pack"])
    model.fit(X, y)

    print('The Model has been created...')

    return model


