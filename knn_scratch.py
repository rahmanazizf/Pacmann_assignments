def knn(X_test,
        X_train, 
        y_train,
        k):
    # cari jarak tiap observasi ke target terdekat
    def euclid_dist(X_test):
        return lambda train_set: ((train_set['x_1'] - X_test['x_1'])**2 + (train_set['x_2'] - X_test['x_2'])**2)**0.5
    
    f_dist = euclid_dist(X_test)
    distances = X_train.apply(f_dist, axis=1)
    
    # cari k tetangga terdekat
    idx_nn = list(np.argsort(distances[0]))[:k]
    y_terdekat = y_train.loc[idx_nn]
    
    # majority vote
    majority_vote = y_terdekat.value_counts(normalize=True).index[0]
    
    return majority_vote, idx_nn
