import lstm
import time
import pandas as pd
from sklearn.metrics import mean_squared_error

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 500
#    seq_len = 50
    mse=[]
    for seq_len in range(5,400,5):
        print('> Loading data... ')
    
        X_train, y_train, X_test, y_test = lstm.load_data(seq_len)
    
        print('> Data Loaded. Compiling...')
    
        model = lstm.build_model([1, 50, 100, 1])
    
        model.fit(
    	    X_train,
    	    y_train,
    	    batch_size=512,
    	    nb_epoch=epochs,
    	    validation_split=0.05)
    
    #    predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    #    lstm.plot_results_multiple(predictions, y_test, 50)
    
    #    predicted = lstm.predict_sequence_full(model, X_test, seq_len)
        predicted = lstm.predict_point_by_point(model, X_test)        
        lstm.plot_results(predicted,y_test)
        mse.append(mean_squared_error(y_test, predicted))

#        print('mean_squared_error : ' ,mean_squared_error(y_test, predicted))
        print('Training duration (s) : ', time.time() - global_start_time)
    df=pd.DataFrame(mse)
    mse.to_csv('mse.csv',index=False)