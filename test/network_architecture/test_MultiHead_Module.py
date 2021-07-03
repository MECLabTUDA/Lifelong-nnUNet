#############################################################################################
# -- Test suite to test the MultiHead Network provided with this extension of the nnUNet -- #
#############################################################################################

import numpy as np
from torch import nn
from typing import Tuple
import os, sys, copy, importlib
from operator import attrgetter
from nnunet_ext.network_architecture import MultiHead_Module
from nnunet_ext.nnunet.network_architecture.generic_UNet import Generic_UNet

# -- Start testing --> This suite only tests the Multi Head Network, whereas no training will be performed -- #
def test_multihead_network():
    # -------------------------#
    # ------ First test ------ #
    # -------------------------#
    # -- Define the splits and task names for the experiments -- #
    splits = ['conv_blocks_context', '     conv_blocks_context    . 0 .    blocks .    1', 'conv_blocks_context.0.blocks.1']
    tasks = ['test', 'new_test', 'test_final']

    # -- Test the three experiments that should work without issues -- #
    for i in range(3):
        if i == 0 or i == 2:
            # -- Perform the first test by using no base model, ie. let the model initialize a GenericUNet -- #
            # -- Initialize a MultiHead Network with an initialized GenericUNet (done by the MultiHead class) -- #
            # -- Look below how the GenericUNet the class initializes should look like -- #
            mh_network = MultiHead_Module.MultiHead_Module(Generic_UNet, splits[i], tasks[i], prev_trainer=None,
                                          input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        else:   # --> Something is wrong here in a second loop
            # -- Perform the second test by using a base model from the first test -- #
            mh_network = MultiHead_Module.MultiHead_Module(Generic_UNet, splits[i], tasks[i], prev_trainer=prev_trainer,
                                          input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter now since we use a base model ..
        
        # -- Check that class object name is correct -- #
        assert mh_network.get_model_type() == 'Generic_UNet',\
            "The class name of the class on which we make a split on is \'{}\' and not \'Generic_UNet\' as expected.".format(mh_network.get_model_type())
        
        # -- Check that the split is correct and that the active task maps as well -- #
        assert mh_network.split == [x.strip() for x in splits[i].split('.')], "The split is \'{}\' and not \'{}\' as expected.".format(mh_network.split[0], splits[i])
        assert mh_network.active_task == tasks[i], "The activated task is \'{}\' and not \'{}\' as expected.".format(mh_network.active_task, tasks[i])

        # -- Check that the split getter is correct as well (especially check that the stripping is correctly done) -- #
        assert mh_network.get_split_path() == '.'.join([x.strip() for x in splits[i].split('.')]),\
            "The split is \'{}\' and not \'{}\' as expected.".format(mh_network.get_split_path(), '.'.join([x.strip() for x in splits[i].split('.')]))

        # -- Check that the number of heads is correct and the name as well -- #
        assert len(mh_network.heads.keys()) == 1, "The number of heads in the module is {} and not {} as expected.".format(len(mh_network.heads.keys()), 1)
        assert list(mh_network.heads.keys())[0] == tasks[i], "The head name/ID is \'{}\' and not \'{}\' as expected.".format(list(mh_network.heads.keys())[0], tasks[i])
        
        # -- Check that the name of the first module in the head is as expected -- #
        # -- Loop through modules based on length of expected split and check if this module is as expected in the head -- #
        try:
          check_split = [x.strip() for x in splits[i].split('.')]
          module_i = copy.deepcopy(mh_network.heads[tasks[i]]) # Will be updated automatically during the loop..
          for index in range(len(check_split)):
            n_i, module_i = list(module_i.named_children())[0]
            # -- Check that the name is equal based on the current index -- #
            assert n_i == check_split[index], "The name of the module at level {} in the head is \'{}\' and not \'{}\' as expected.".format(index, n_i, check_split[index])
        except:
          assert False, "An error occured by trying to check if the head starts with the expected module, which should have worked if the split worked as expected"

        # -- Initialize one new task and perform a bunch of tests -- #
        mh_network.add_new_task('single_addition')

        # -- Check that the number of tasks changed and the keys are as desired -- #
        assert len(mh_network.heads.keys()) == 2, "The number of heads in the module is {} and not {} as expected.".format(len(mh_network.heads.keys()), 2)
        
        # -- Check that the name of the task we just added exists the way we expect -- #
        try:
            _ = mh_network.heads['single_addition']
        except:
            assert False, "The desired task \'single_addition\' does not exist in the head althoug it should."
        
        # -- Check that the activated task changed after the addition of a new task -- #
        assert mh_network.active_task == tasks[i],\
            "The activated task \'{}\' is wrong and should not have changed after creation of a new task.".format(mh_network.active_task)
        
        # -- Check that the state_dicts are identically for every task in the head, since we did not perform any train -- #
        for key in mh_network.heads[tasks[i]].state_dict():
            assert (np.array(mh_network.heads[tasks[i]].state_dict()[key].tolist()) == np.array(mh_network.heads['single_addition'].state_dict()[key].tolist())).all(),\
                "The state_dicts in \'{}\' and \'single_addition\' with key \'{}\' are not identical although they should.".format(tasks[i], key)
        
        # -- Check that when assembling a model, the correct task is activated -- #
        _ = mh_network.assemble_model('single_addition')
        assert mh_network.active_task == 'single_addition', "After the model fusion, the activated task should be updated as well which is not the case."
        
        # -- Check that the state dicts for every assembled model is equal since we did not train anything -- #
        model_1 = mh_network.assemble_model(tasks[i])
        model_2 = mh_network.assemble_model('single_addition')
        for key in model_1.state_dict():
            assert (np.array(model_1.state_dict()[key].tolist()) == np.array(model_2.state_dict()[key].tolist())).all(),\
                "The state_dicts in the activated models with key \'{}\' are not identical although they should.".format(key)
        del model_1
        
        # -- Activate the same task that is active again and check that nothing changed -- #
        model_1 = mh_network.assemble_model('single_addition')
        assert mh_network.active_task == 'single_addition', "After the model fusion, the activated task should be updated as well which is not the case."
        for key in model_1.state_dict():
            assert (np.array(model_1.state_dict()[key].tolist()) == np.array(model_2.state_dict()[key].tolist())).all(),\
                "The state_dicts after activating a model twice should not change, but they are different for key \'{}\'.".format(key)
        del model_1, model_2
        
        # -- Initialize 3 new tasks and check that everything is as expected -- #
        mh_network.add_n_tasks_and_activate(['test_1', 'test_2', 'test_3'], 'test_2')
        assert len(mh_network.heads.keys()) == 5, "The number of heads in the module is {} and not {} as expected.".format(len(mh_network.heads.keys()), 5)
        assert mh_network.active_task == 'test_2', "After activating a task the activated task should be updated as well when adding n tasks at once."
        # -- Just check if 'test_2' head and 'single_addition' head have same state_dicts --> Everything else is than equal as well -- #
        for key in mh_network.heads['test_2'].state_dict():
            assert (np.array(mh_network.heads['test_2'].state_dict()[key].tolist()) == np.array(mh_network.heads['single_addition'].state_dict()[key].tolist())).all(),\
                "The state_dicts in \'test_2\' and \'single_addition\' with key \'{}\' are not identical although they should.".format(key)
        
        # -- Replace conv2d layer in a specific head -- #
        mh_network.replace_layers(mh_network.heads['test_1'], nn.Conv2d, nn.AdaptiveMaxPool2d((6, 6)))
        
        if i == 0:
            # -- Check that the conv2d layers in this specific head changed and that the a different head is unchanged -- #
            assert isinstance(attrgetter('conv_blocks_context.1.blocks.0.conv')(mh_network.heads['test_1']), type(nn.AdaptiveMaxPool2d(6, 6))),\
                "The replacing of layers unfortunately did not work."
            assert isinstance(attrgetter('conv_blocks_context.1.blocks.0.conv')(mh_network.heads[tasks[i]]), type(nn.Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1]))),\
                "The replacing of layers unfortunately had also effects on parts where it should not have."
        else:
            # -- Check that the conv2d layers in this specific head changed and that the a different head is unchanged -- #
            assert isinstance(attrgetter('conv_blocks_context.0.blocks.1.conv')(mh_network.heads['test_1']), type(nn.AdaptiveMaxPool2d(6, 6))),\
                "The replacing of layers unfortunately did not work."
            assert isinstance(attrgetter('conv_blocks_context.0.blocks.1.conv')(mh_network.heads[tasks[i]]), type(nn.Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1]))),\
                "The replacing of layers unfortunately had also effects on parts where it should not have."
        
        # -- Create a backup, otherwise there will be a mixup in the next iteration -- #
        prev_trainer = mh_network.model
        del mh_network
    del prev_trainer

    # ------------------------- #
    # ------ Second test ------ #
    # ------------------------- #
    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Check if a split is correcty simplified by the model -- #
    try:
        split = 'conv_blocks_localization  . 2  . 0.blocks.0'
        mh_network = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                                      input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        assert mh_network.get_split_path() == 'conv_blocks_localization.2', "The split should have been clipped to the minimal part but it has not been."
    except Exception as e:
        raise e

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Check that the split on the very last layer is working -- #
    try:
        split = 'seg_outputs.2'
        mh_network = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                                      input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        check_split = [x.strip() for x in split.split('.')]
        module_i = copy.deepcopy(mh_network.heads['test']) # Will be updated automatically during the loop..
        for index in range(len(check_split)):
          n_i, module_i = list(module_i.named_children())[0]
          # -- Check that the name is equal based on the current index -- #
          assert n_i == check_split[index], "The name of the module at level {} in the head is \'{}\' and not \'{}\' as expected.".format(index, n_i, check_split[index])
    except Exception as e:
        if not isinstance(e, AssertionError):
            assert False, "An error occured by trying to check if the head starts with the expected module, which should have worked if the split worked as expected"
        raise e

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Test if the split on the deepest layer works -- #
    try:
        split = 'conv_blocks_localization.2.0.blocks.0.instnorm'
        mh_network = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                                      input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        check_split = [x.strip() for x in split.split('.')]
        module_i = copy.deepcopy(mh_network.heads['test']) # Will be updated automatically during the loop..
        for index in range(len(check_split)):
          n_i, module_i = list(module_i.named_children())[0]
          # -- Check that the name is equal based on the current index -- #
          assert n_i == check_split[index], "The name of the module at level {} in the head is \'{}\' and not \'{}\' as expected.".format(index, n_i, check_split[index])
    except Exception as e:
        if not isinstance(e, AssertionError):
            assert False, "An error occured by trying to check if the head starts with the expected module, which should have worked if the split worked as expected"
        raise e

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Do some tests where an error should be thrown -- #
    # -- Give network a non string split path -- #
    # -- Use RuntimeErrors to be able to distinguish if the code run through until the end which it shouldn't --> then catch it -- #
    try:
        split = ['conv_blocks_localization', '2', '0']
        _ = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                             input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        raise RuntimeError
    except Exception:
        if isinstance(Exception, RuntimeError):
          assert False, "When providing a non like split, it is expected that the model should throw an error but it did not."
          
    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Give network a direct split to first layer and indirect to first layer and expect an error thrown -- #
    for split in ['conv_blocks_localization', 'conv_blocks_localization.0', 'conv_blocks_localization.0.0.blocks', 'conv_blocks_localization.0.0.blocks.0.conv']:
        try:
            _ = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                                 input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
            raise RuntimeError
        except Exception:
            if isinstance(Exception, RuntimeError):
              assert False, "When trying to split on first layer, the model should throw an error since this makes no sense but it did not."
            
    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Give network a wrong split path that does not exist -- #
    try:
        split = 'conv_blocks_localization.8.0'
        _ = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=None,
                             input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        raise RuntimeError
    except Exception:
        if isinstance(Exception, RuntimeError):
            assert False, "When providing a non existing split path, it is expected that the model should throw an error but it did not."

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Give network from wrong, non nn.Module module (class_object) -- #
    try:
        split = 'conv_blocks_localization.1.0'
        _ = MultiHead_Module.MultiHead_Module(Tuple, split, 'test', prev_trainer=None,
                             input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        raise RuntimeError
    except Exception:
        if isinstance(Exception, RuntimeError):
            assert False, "The split Network should only be able for models based on nn.Module, but it did not."

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Use a prev_trainer that is not of the same type as the specified class_object -- #
    try:
        split = 'conv_blocks_localization  . 2  . 0.block.0'
        _ = MultiHead_Module.MultiHead_Module(Generic_UNet, split, 'test', prev_trainer=nn.Module(),
                             input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        raise RuntimeError
    except Exception:
        if isinstance(Exception, RuntimeError):
            assert False, "When providing a prev_trainer, that is different from the specified class_object an error is expected."    

    # -- Reload the Multi Head module, otherwise there will be mixups during runtime and wrong results will be produced -- #
    importlib.reload(MultiHead_Module)
    
    # -- Try to use an empty prev_trainer that maps the class_object -- #
    try:
        split = 'conv_blocks_localization  . 2  . 0.block.0'
        _ = MultiHead_Module.MultiHead_Module(nn.Module, split, 'test', prev_trainer=nn.Module(),
                             input_channels=3, base_num_features=5, num_classes=2, num_pool=3) # Does not matter how to set it, just testing ..
        raise RuntimeError
    except Exception:
        if isinstance(Exception, RuntimeError):
            assert False, "When providing a prev_trainer, it is expected to throw an error if the trainer is empty."
    

if __name__ == "__main__":
    # -- Block all prints that are done during testing which are no errors but done in calling functions -- #
    sys.stdout = open(os.devnull, 'w')

    # -- Run the test suite -- #
    test_multihead_network()


""" GenericUNet using input_channels=3, base_num_features=5, num_classes=2, num_pool=3:
Generic_UNet(
  (conv_blocks_localization): ModuleList(
    (0): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (1): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (2): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (conv_blocks_context): ModuleList(
    (0): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (1): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (2): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (3): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 40, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (dropout): Dropout2d(p=0.5, inplace=True)
            (instnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (dropout): Dropout2d(p=0.5, inplace=True)
            (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (td): ModuleList(
    (0): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (tu): ModuleList(
    (0): Upsample()
    (1): Upsample()
    (2): Upsample()
  )
  (seg_outputs): ModuleList(
    (0): Conv2d(10, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): Conv2d(5, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): Conv2d(5, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)
"""