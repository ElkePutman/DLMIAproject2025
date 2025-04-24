from classfication_functions import ClassificationACDC
from randomforrestclassifier import RandomForrestACDC

def main():
    # Put the right names for the paths leading with the corresponding data 
    train_gt_path = r"INSERT PATH TO GROUND TRUTH MAKS HERE "                                # path to the folder with the ground truth masks of the training data set, of both the ED and ES phase 
    train_info_path = r"INSERT PATH TO PATIENT INFO OF TRAIN GROUP HERE "                    # path to the folder with the files with patient information from the train data set
    test_segmentation_path = r"INSERT PATH TO SEGMENTATION MASKS YOU WANT TO CLASSIFY HERE with" # path to the folder with the segmentation masks of the test data set, of both the ED and ES phase 
    test_info_path = r"INSERT PATH TO PATIENT INFO OF TEST GROUP HERE "                      # path to the folder with the files with patient information from the test data set

    # Indicate if you want to train the model and if you want to get a ROC curve
    train_model = False         # True: training, False: using saved model
    roc = True                  # True: ROC curve, False: no ROC curve

    # prepare the training data
    data_loader = ClassificationACDC()
    data_loader.build_class_dict_ACDC(train_gt_path, train_info_path, 'train')
    train_patients = data_loader.create_dict_cardiac_info('train')
    
    # prepare the test data 
    data_loader.build_class_dict_ACDC(test_segmentation_path, test_info_path, 'test')
    test_patients = data_loader.create_dict_cardiac_info('test')
    print('hallo')

    # set-up the randomforrest classifier 
    rf_model = RandomForrestACDC(test_patients, train_patients)

    # train the random forrest or use the saved model
    if train_model:
        rf_model.trainer()
    else:
        rf_model.load_model()

    # test the random forrest 
    accuracy = rf_model.tester()

    if roc:
        rf_model.roc_curve()

    
    

if __name__ == "__main__":
    main()