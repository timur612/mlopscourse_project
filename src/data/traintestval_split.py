import os
import random
import shutil

random.seed(42)


def traintestval_split(
    input_data_path_seal: str, input_data_path_no_seal: str, output_data_path: str
):
    """split image dataset into train, test and val
    args:
        input_data_path_seal - path for folder with img dataset seal
        input_data_path_no_seal - path for folder with img dataset NO seal
        output_data_path - path for splitted img dataset
    """

    train_folder = os.path.join(output_data_path, "train")
    val_folder = os.path.join(output_data_path, "eval")
    test_folder = os.path.join(output_data_path, "test")

    imgs_list_seal = [filename for filename in os.listdir(input_data_path_seal)]
    imgs_list_no_seal = [filename for filename in os.listdir(input_data_path_no_seal)]

    random.shuffle(imgs_list_seal)
    random.shuffle(imgs_list_no_seal)

    train_size_seal = int(len(imgs_list_seal) * 0.70)
    val_size_seal = int(len(imgs_list_seal) * 0.15)
    test_size_seal = int(len(imgs_list_seal) * 0.15)

    train_size_no_seal = int(len(imgs_list_no_seal) * 0.70)
    val_size_no_seal = int(len(imgs_list_no_seal) * 0.15)
    test_size_no_seal = int(len(imgs_list_no_seal) * 0.15)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path + "/seal")
            os.makedirs(folder_path + "/no_seal")

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list_seal):
        if i < train_size_seal:
            dest_folder = train_folder
        elif i < train_size_seal + val_size_seal:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(
            os.path.join(input_data_path_seal, f),
            os.path.join(dest_folder + "/seal/", f),
        )

    for i, f in enumerate(imgs_list_no_seal):
        if i < train_size_no_seal:
            dest_folder = train_folder
        elif i < train_size_no_seal + val_size_no_seal:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(
            os.path.join(input_data_path_no_seal, f),
            os.path.join(dest_folder + "/no_seal/", f),
        )
