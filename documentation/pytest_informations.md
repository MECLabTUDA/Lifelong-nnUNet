# PyTest Informations

The [test folder](https://github.com/camgbus/Lifelong-nnUNet/tree/continual_learning/test) on the CL branch contains multiple PyTests that are used to validate the correctness of the provided Source Code. In general, we provide 5 tests, whereas four of those use at least one GPU. The only test not using a GPU is the one testing the functionality to change mask labels for dataset based on a mapping file as described [here](change_mask_labels.md).

### PyTests for Multi-Head and CL Trainers

The tests located [here](https://github.com/camgbus/Lifelong-nnUNet/tree/continual_learning/test/training/network_training) test all provided Trainers that are related to the CL branch and Multi-Head Module. Before using the CL extension for training or as a foundation for something else, we recommend to run those tests at the beginning to ensure that everything is as expected; the tests should all terminate without errors. Given the complexity and large test suits, the tests take several hours to complete. Knowing this, one can easily remove the tests once they are sucessfully run for the first time, since they do not need to be executed every time as long as no major changes have been made. During those tests, a log file is created within the folder of the trained models located in the `RESULTS_FOLDER`.
