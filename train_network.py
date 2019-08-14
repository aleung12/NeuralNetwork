from neural_network import NeuralNetwork
from predict_claims import TrainedNeuralNetwork
from load_data import load_data
import os, time, datetime
import numpy as np



def start(train_data, test_data, resuming=False, testing=True):

    nn = NeuralNetwork(train_data[:,:-1], train_data[:,-1].reshape(-1,1)/800)

    if resuming: 
        si, last_lossf, nn.learning_rate = np.load('state/state.npy')
        nn.weights0 = np.load('state/weights0.npy')
        nn.weights1 = np.load('state/weights1.npy')
        nn.weights2 = np.load('state/weights2.npy')
        nn.biasw1   = np.load('state/biasw1.npy')
        nn.biasw2   = np.load('state/biasw2.npy')
        best_test_lossf = np.load('state/best/test_lossf.npy')
    else:
        if not os.path.exists('state'): os.makedirs('state')
        if not os.path.exists('state/best'): os.makedirs('state/best')
        best_test_lossf = np.inf
        si = 0
        
    out = open('track_info.dat', 'a')

    it = 2000
    for i in range(int(si), it):
        print('\n\niteration %d of %d\n' % (i+1, it))
        print('    layer 1:    n = %d' % nn.n1)
        print('    layer 2:    n = %d' % nn.n2)
        
        t0 = time.time()
        print('\nbegin feed_forward() at ' + datetime.datetime.now().strftime("%H:%M:%S on %d %B %Y"))

        nn.feed_forward()

        print('\n                               first                   third              ')
        print('                    min       quartile     median     quartile      max     ')
        print('                 ----------  ----------  ----------  ----------  ---------- ')
        print('    y          |%11s %11s %11s %11s %11s' % quartiles(nn.y))
        print('    output     |%11s %11s %11s %11s %11s' % quartiles(nn.output))
        print('')
        print('    weights0   |%11s %11s %11s %11s %11s' % quartiles(nn.weights0))
        print('    weights1   |%11s %11s %11s %11s %11s' % quartiles(nn.weights1))
        print('    weights2   |%11s %11s %11s %11s %11s' % quartiles(nn.weights2))
        print('')
        print('    biasw1     |%11s %11s %11s %11s %11s' % quartiles(nn.biasw1))
        print('    biasw2     |%11s %11s %11s %11s %11s' % quartiles(nn.biasw2))
        
        te = time.time()-t0
        print('\nfeed_forward() took %dm %.2fs\n' % (te//60, te%60))

        lossf = nn.loss_function()
        print('\n    loss function (training):     %.9f\n' % lossf)
        out.write(datetime.datetime.now().strftime("%H:%M:%S %D") + ' %2s %6s  %e  %.6f' % (k, i+1, nn.learning_rate, lossf))
        
        if i > 10:
            if last_lossf > lossf: 
                nn.save_weights()
                np.save('state/state.npy', [i, lossf, nn.learning_rate])
                if i < 200: nn.learning_rate *= 1.01
                else:       nn.learning_rate *= 1.005
            else:
                nn.learning_rate /= 1.007

            tn = TrainedNeuralNetwork(test_data[:,:-1])

            tn.load_weights((nn.weights0, nn.weights1, nn.weights2, nn.biasw1, nn.biasw2))
            tn.feed_forward()

            test_lossf = np.mean(np.square(test_data[:,-1].reshape(-1,1) - 800*tn.output))
            print('    loss function (validation):   %.3f HKD^2 -> %.3f HKD (rms)\n' % (test_lossf, np.sqrt(test_lossf)))
            out.write('  %.3f' % np.sqrt(test_lossf))

            if best_test_lossf > test_lossf:
                best_test_lossf = test_lossf
                np.save('state/best/test_lossf.npy', best_test_lossf)
                nn.save_weights(best=True)

        t0 = time.time()
        print('\nbegin back_prop() at ' + datetime.datetime.now().strftime("%H:%M:%S on %d %B %Y"))
        print('\n    learning rate = %e' % nn.learning_rate)

        nn.back_prop(testing)
        
        if testing:
            print('\n    weights0   |%11s %11s %11s %11s %11s' % quartiles(nn.weights0))
            print('    weights1   |%11s %11s %11s %11s %11s' % quartiles(nn.weights1))
            print('    weights2   |%11s %11s %11s %11s %11s' % quartiles(nn.weights2))
            print('    biasw1     |%11s %11s %11s %11s %11s' % quartiles(nn.biasw1))
            print('    biasw2     |%11s %11s %11s %11s %11s' % quartiles(nn.biasw2))
        
        te = time.time()-t0
        print('\nback_prop() took %dm %.2fs\n' % (te//60, te%60))
            
        last_lossf = lossf
        out.write('\n')
    out.close()


## returns quartiles
quartiles = lambda x : ('%.3e'%np.min(x), '%.3e'%np.percentile(x,25), '%.3e'%np.median(x), '%.3e'%np.percentile(x,75), '%.3e'%np.max(x))



if __name__ == '__main__':

    ## Provide path to flight delays data
    nn_input, is_claim = load_data('flight_delays_data.csv')

    ## Resuming?
    resuming = False
    k_in_progress = 1

    ## For scikit-learn k-fold cross-validation
    from sklearn.model_selection import KFold
    import sys

    ## Split data into training and validation sets
    kfold = KFold(n_splits=max(int(sys.argv[1]), 2), shuffle=True, random_state=0)

    k = 1
    data = np.concatenate((nn_input.T, is_claim.T)).T
    for train, test in kfold.split(data):

        if resuming and k < k_in_progress:
            k += 1
            continue

        if not resuming:
            ## Train neural network from scratch
            start(data[train], data[test])

        else:
            ## Resume training from state saved in ./state/
            start(data[train], data[test], resuming)

        if not os.path.exists('state/best/fold%d'%k): os.makedirs('state/best/fold%d'%k)
        os.system('cp -p state/best/*.npy state/best/fold%d/' % k)

        ## When advancing to the next k-fold, henceforth not resuming
        resuming = False
        k += 1
