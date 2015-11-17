import prep_data
import w2v_train_NN
import test_on_train_data

prep_data.main()
w2v_train_NN.main()
test_on_train_data.main(True)