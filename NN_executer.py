import prep_data
import w2v_train_NN
import test_on_train_data

print('prep_data...')
prep_data.main()
print('w2v_train_NN...')
w2v_train_NN.main()
print('test_on_train_data...')
test_on_train_data.main(train = True)