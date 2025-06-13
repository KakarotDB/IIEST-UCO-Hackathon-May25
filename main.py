from app.gui import run_gui
from app.server_testcode import predict_user_cluster
from train.model1 import train_model1
from train.model2 import train_model2
from train.model3 import train_model3
from train.final_model import final_model
from train.server_final_model import train_server_model_kmeans

if __name__ == "__main__":
    # train_model1()
    # train_model2()
    # train_model3()
    # final_model()
    train_server_model_kmeans()

    # predict_user_cluster()
    # run_gui()