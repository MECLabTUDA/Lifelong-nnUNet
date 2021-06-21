#########################################################################################################
#-------------------This class represents the generic nnU-Net enabling multiple heads.------------------#
#########################################################################################################

import copy
from torch import nn
from operator import attrgetter
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import Generic_UNet, ConvDropoutNormNonlin


class MH_Generic_UNet(nn.Module):
    r"""This class is used for nnU-Nets that have multiple heads using a shared body.
        The heads are stored in a ModuleDict, whereas the task name is the key to the
        corresponding head/module.
    """
    def __init__(self, split_at, task, data_parallel, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, prev_trainer=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        r"""Constructor of the generic nnU-Net for multiple heads. When a Generic_UNet is provided as prev_trainer
            this class will be initialized using the attributes from this class. The split will then be performed
            on the provided model, but the model will not be touched since a new MH_Generic_UNet class will be created.
            data_parallel is used in order to fuse body and head as nn.DataParallel, if it is set to False, self.model
            representing the body with the current task is a nn.Module. It should be based on the module the use_base is/
            the model that will be initialized. The nnU-Net for instance expects either a SegmentationNetwork or DataParallel
            to perform .forward on the network, so in this case data_parallel should be always True.
            NOTE: prev_trainer is not necessary but can be used when a pre-trained model should be used as initialization
                  and not a complete new initialized nnUNet. Further, all splits that will be added using add_empty_module
                  use the Module that is extracted due to the split from the prev_trainer model, including its state_dict etc.
                  When providing a new Module to add to the Network using add_module etc. ensure that all layers are there
                  that the Generic_UNet is using during training.
        """
        # -- Initialize using super -- #
        super().__init__()

        # -- If no model is provided, initialize one -- #
        if prev_trainer is None:
            # -- Initialize a conventional generic nnU-Net -- #
            self.model = Generic_UNet(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                      feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                      nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization, final_nonlin, weightInitializer,
                                      pool_op_kernel_sizes, conv_kernel_sizes, upscale_logits, convolutional_pooling,
                                      convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        else:
            # -- Define the classes that are allowed given data_parallel -- #
            if data_parallel:
                # -- prev_trainer can only be Generic_UNet or nn.DataParallel but nothing else -- #
                classes_permitted = (Generic_UNet, nn.DataParallel)
            else:
                # -- prev_trainer can only be Generic_UNet or nn.Module but nothing else -- #
                classes_permitted = (Generic_UNet, nn.Module)
            # -- Ensure that prev_trainer is a Generic_UNet, otherwise this will not work -- #
            assert isinstance(prev_trainer, classes_permitted),\
                "This function transform a Generic_UNet to MH_Generic_UNet, so provide a Generic_UNet and not a {} object.".format(type(prev_trainer))
            # -- Set the models class to prev_trainer -- #
            self.model = prev_trainer

            """
            # -- Get all attributes (excluding functions and anything else) the provided prev_trainer --> set the same in self -- #
            attributes = [a for a in dir(prev_trainer) if not a.startswith('__') and not callable(getattr(prev_trainer, a))]
            # -- Loop through these attributes and set self accordingly -- #
            for attribute in attributes:
                # -- Set self accordingly -- #
                # -- NOTE: Since self inherits Generic_UNet and prev_trainer is of the class Generic_UNet -- #
                # --       the functions from Generic_UNet still work --> just copied the attributes -- #
                # --       were copied, the functions use self... and thus the right attributes :) -- #
                setattr(self, attribute, getattr(prev_trainer, attribute))
            """

        # -- Save the split_at, so it can be used throughout the class -- #
        assert isinstance(split_at, str), "The provided split needs to be a string.."
        self.split = [x.strip() for x in split_at.split('.')] # --> Never know what users input so strip layer names

        # -- Set flag if DataParallel or Module should be used in self.model -- #
        self.data_parallel = data_parallel

        # -- Change self.split if necessary --> Remove redundant split names to shorten it to bare minimum -- #
        # -- Check that last element of split does not refer to first element of last splits element -- #
        # -- since this can be omitted --> If this is the case remove the last split name and print a message -- #
        simplify = len(self.split) > 1  # Otherwise it makes no sense and the while loop results in an error
        split = self.split[:]
        while simplify:
            first_layer_name, _ = next(attrgetter('.'.join(self.split[:-1]))(self.model).named_children())
            if self.split[-1] == first_layer_name:
                # -- Remove the last split element since this leads to an error due to wrong interpretation by user -- #
                new_split = self.split[:-1]
                # -- Print a Note -- #
                print('Note: The split \'{}\' has been transformed to \'{}\' since it specifies the same split.\n'.format('.'.join(self.split), '.'.join(new_split)))
                # -- Update self.split
                self.split = new_split
                del new_split   # Not necessary anymore
                simplify = len(self.split) > 1
            else:
                simplify = False

            # -- Assert if the split is empty -- #
            assert not(len(self.split) == 1 and simplify), "The provided split \'{}\' is empty after simplification and would split before the first layer..".format('.'.join(split))
        del split # Not necessary anymore

        # -- Set flag if DataParallel or Module should be used in self.model -- #
        self.data_parallel = data_parallel
        
        # -- Define empty ModuleDict for all the heads -- #
        self.heads = nn.ModuleDict()

        # -- Define a variable that specifies the active_task -- #
        assert isinstance(task, (str, int)), "The provided task needs to be an integer (ID) or string, not {}..".format(type(task))
        self.active_task = task

        # -- Split self.model based on split_at and store the init_module -- #
        # -- NOTE: If prev_trainer was used, init_module is now the splitted part from -- #
        # --       the prev_trainer and not from a new generated nnUNet, ie. this part -- #
        # --       is always used when calling self.add_empty_module --> keep this in mind -- #
        self.body, self.init_module, _ = self._split_model_recursively_into_body_head(layer_id=0, model=self.model) # Start from root with full model

        # -- Register the init_module into self.heads based on self.active_task
        self.add_new_task(self.active_task)

        # -- Assemble the model so it can be used for training -- #
        self.model = self._assemble_model(task)
        #print(self.replace_layers(self.heads[task], nn.Conv2d, nn.AdaptiveMaxPool2d((6, 6)))) 

    def forward(self, x):
        r"""Forward pass during training --> task needs to be specified before calling forward.
            Assemble the model based on self.body and selected task. Then use parent class to perform
            forward pass. Follow with a split of the model after the pass to update the head and the body.
        """
        # -- Let the Generic_UNet class do the work since the assembled model is a generic nnU-Net -- #
        res = Generic_UNet.forward(self.model, x) # --> Do not use super, since we want to set the correct self object ;)

        # -- Update the body and corresponding head so these variables are always up to date -- #
        # -- Simply do a splitting again -- #
        self.body, self.heads[self.active_task], _ = self._split_model_recursively_into_body_head(layer_id=0, model=self.model)

        # -- Return the forward result generated by Generic_UNet.forward -- #
        return res

    def get_split_path(self):
        r"""This function returns the path to the layer, where the split has been performed.
        """
        # -- Return the split path -- #
        return join_texts_with_char(self.split, '.')

    def add_new_task(self, task):
        r"""Use this function to add the initial (empty) module from nnU-Net based on the split.
            Specify the task name with which it will be registered in the ModuleDict.
            NOTE: If the task already exists, it will be overwritten.
        """
        # -- Create a new task in self.heads with the initialized module which is from nnU-Net itself -- #
        self.heads[task] = copy.deepcopy(self.init_module)
    
    def add_n_tasks_and_activate(self, list_of_tasks, activate_with):
        r"""Use this function to initialize for each task name in the list a new head. -- > Important when restoring a model,
            since the heads are not created in this init function and need to be set manually before loading a state_dict
            that includes n heads. Further self.model will be assembled based on activate_with. In the case of calling
            this function before restoring, the user needs to provide the correct activate_with, ie. the head that was activated
            in times of saving the Multi Head Network.
        """
        # -- Loop through list of tasks -- #
        for task in list_of_tasks:
            # -- Add the task to the head -- #
            self.add_new_task(task)

        # -- Assemble the model based on activate_with -- #
        self._assemble_model(activate_with)

    def replace_layers(self, model, old, new):
        r"""This function is based on: https://www.kaggle.com/ankursingh12/why-use-setattr-to-replace-pytorch-layers.
            It can be used to replace a desired layer in the provided model with a new Module.
            :param model: nn.Module in which the layer (old) should be replaced (with new)
            :param old: nn.Module representing a specific layer, like nn.Conv2d, nn.ReLU, etc.
            :param new: nn.Module representing a specific layer, like nn.Conv2d, nn.ReLU, etc. which will replace old
            :return: Function returns the updated module.
            NOTE: This function can be used for instance to replace e.g. a desired layer in the body/head. It does
                  not have to be Module directly from torch.nn, it can also be any Module that inherits nn.Module.
                  For instance we could replace a nnU-Net seg_outputs layer in a specified head with a new desired layer.
                  For this, the model would be the desired head (nn.Module) not a ModuleDict.
        """
        # -- Check that the types are as desired -- # 
        assert model is not None and new is not None and old is not None, "To replace a Module, the layers need to be Modules as well as the model.."
        
        # -- Loop through model children -- #
        for name, module in model.named_children():
            # -- If there are still children left, go recursive -- #
            if len(list(module.children())) > 0:
                # -- Go one layer deeper -- #
                self.replace_layers(module, old, new)

            # -- If the module is an instance of old, replace it with new -- #
            if isinstance(module, old):
                # -- Replace -- #
                setattr(model, name, new)

        # -- Return the updated module -- #
        return model

    """ Do not include since this makes the module vulnerable when user does not know what he is doing..
    def add_module(self, module, task):
        rUse this function to add a module specified to the heads given a task name.
            NOTE: If the task exists, it will be overwritten.
        
        # -- Check that the provided module is a Module -- #
        assert isinstance(module, nn.Module), "Module is not an nn.Module"

        # -- Create a new task using the provided module -- #
        self.heads[task] = module

    def add_module_copy(self, module, task):
        rUse this function to add the copy of a model to the heads given a task name,
            but not the model directly.
            NOTE: If the task exists, it will be overwritten.
        
        # -- Check that the provided module is a Module -- #
        assert isinstance(module, nn.Module), "Module is not an nn.Module"

        # -- Create a new task using a deepcopy of the provided module -- #
        self.heads[task] = copy.deepcopy(module)"""

    def _split_model_recursively_into_body_head(self, layer_id, model, body=nn.Module(), head=nn.Module(),
                                                parent=list(), simplify_split=False):
        r"""This function splits a provided model into a base and a single head.
            It returns the body, head and layeron_id on which the split is performed.
            :param layer_id: The current index of the layer, so 0 is root, 1 is child of root, etc.
            :param model: The model to be split or a submodule of it
            :param body: Representing the base of the initial model since this is a recursive function
            :param head: Representing the head after the split of the initial model since this is a recursive function
            :param parent: String representing the name of the parent module
            :param simplify_split: Bool whether the split should be simplified. Use this only if the function is called from outside.
                                   When it is called from inside, self.split is already simplified, but if it should change throughout
                                   it should be simplified again, then set this to true (wondering if that's ever the case..).
            :return body, head, layer_id: The splitted model (body and head) with the layer_id where the split is performed
            NOTE: This function is a recursive function, a split should be initialized with self.model and
                  a layer_id of 0. Further, the returned layer_id is for recursion and might not be of interest
                  to the user since at the end it is equal to the length of self.split.
        """
        # -- If the function is called the first time, perform some checks -- #
        if layer_id == 0:
            # -- Assert if the split is empty -- #
            assert len(self.split) != 0, "The provided split is empty.."

            first_layer = next(model.named_children()) # Extract the first layer in the model
            # -- Assert if the user tries to split at very first layer --> makes no sense, use original nnU-Net instead -- #
            assert not(layer_id == 0 and self.split[layer_id] == first_layer[0] and len(self.split) == 1),\
                "You tried to split before the first layer, so the body would be empty --> body can never be empty.."
            # -- Test if the provided path even exists -- #
            try:
                # -- Extract the part where the split will be and check if it even exists -- #
                _ = attrgetter('.'.join(self.split))(model)
            except:
                assert False, "The provided split path \'{}\' does not exist..".format('.'.join(self.split))

            # -- Change self.split if desired --> Remove redundant split names to shorten it to bare minimum -- #
            # -- Check that last element of split does not refer to first element of last splits element -- #
            # -- since this can be omitted --> If this is the case remove the last split name and print a message -- #
            if simplify_split:
                simplify = len(self.split) > 1  # Otherwise it makes no sense and the while loop results in an error
                split = self.split[:]
                while simplify:
                    first_layer_name, _ = next(attrgetter('.'.join(self.split[:-1]))(model).named_children())
                    if self.split[-1] == first_layer_name:
                        # -- Remove the last split element since this leads to an error due to wrong interpretation by user -- #
                        new_split = self.split[:-1]
                        # -- Print a Note -- #
                        print('Note: The split \'{}\' has been transformed to \'{}\' since it specifies the same split.\n'.format('.'.join(self.split), '.'.join(new_split)))
                        # -- Update self.split
                        self.split = new_split
                        del new_split   # Not necessary anymore
                        simplify = len(self.split) > 1
                    else:
                        simplify = False

                    # -- Assert if the split is empty -- #
                    assert not(len(self.split) == 1 and simplify), "The provided split \'{}\' is empty after simplification and would split before the first layer..".format('.'.join(split))
                del split # Not necessary anymore

        # -- Perform splitting: Loop through the models modules -- #
        for idx, (name, n_module) in enumerate(model.named_children()):
            # -- Deep copy the module to bypass the runtime error that would occur due to changing of the module -- #
            module = copy.deepcopy(n_module)
            # -- If, given layer_id, a desired module from the path is not reached and layer_id is -- #
            # -- smaller than number of split path elements --> set body -- #
            # -- Ensure that nothing is added twice due to recursion, thus layer_id == 0 -- #
            if layer_id == 0 and layer_id < len(self.split)-1 and name != self.split[layer_id]:
                # -- Add this whole module with all children to body -- #
                body.add_module(name, module)

            # -- If the layer_id is equal to the number of split path elements and equal to the last element in split -- #
            # -- the split point is found --> the last node of the split, so everything after that is head -- #
            elif layer_id == len(self.split)-1:
                if name == self.split[layer_id]:
                    # -- Extract the module in which the split is performed -- #
                    if layer_id == 0:   # Split on first layer, no parent exists --> parent is model
                        parent_mod = model
                    else:   # Deeper layer, parent is set by now and in the body
                        parent_mod = attrgetter('.'.join(parent))(body)

                    # -- Extract the children with their names -- #
                    children = list(parent_mod.named_children())
                    # -- Determine the part that updates the body considered the split -- #
                    body_part = children[:idx]
                    # -- Determine the part that belongs to the head -- #
                    head_part = children[idx:]

                    # -- Only add/replace body_part if it is not an empty Module -- #
                    if len(body_part) != 0:
                        # -- Update the body -- #
                        if len(parent) == 0:    # Body is likely to be empty, so add every model, e.g. layer_id 0 and split on same layer
                            # -- Loop through body_parts and add them to the body -- #
                            for name_part, part in body_part:
                                body.add_module(name_part, part)
                        else:
                            # -- Loop through head_parts and delete them in the body -- #
                            for name_part, _ in head_part:
                                delattr(attrgetter('.'.join(parent))(body), name_part)
                    else:
                        # -- Delete the module from body part specified by the path from parent and the name -- #
                        delattr(attrgetter('.'.join(parent))(body), name)
                    
                    # -- Add the module to the head with the given name -- #
                    if len(head_part) != 0:
                        # -- In this step we are in recursion level, so only add the current module, -- #
                        # -- due to loop the rest of the sibling will be added as well and due to recursion -- #
                        # -- all aunts, uncles, etc. will be added as well -- #
                        #head.add_module(head_part[0][0], head_part[0][1])
                        # -- In this step we are in recursion level, so add all the siblings of the current module, -- #
                        # -- due to loop all aunts, uncles, etc. will be added as well after that -- #
                        # -- Loop through head_part and add them to the head -- #
                        for name_part, part in head_part:
                            head.add_module(name_part, part)

                    # -- Increase layer_id so all other modules are added to head and removed from body -- #
                    layer_id += 1

                # -- Do nothing else, since the part is already in the body and does not belong to the head -- #

            # -- If layer_id is greater than number of split path elements --> set head -- #
            elif layer_id > len(self.split)-1:
                # -- Add this whole module with all children to head -- #
                head.add_module(name, module)
                # -- Try to remove the module from the body when adding everything to the head -- #
                try:
                    # -- Try to delete the module from body part specified by the path from parent and the current name -- #
                    delattr(attrgetter('.'.join(parent))(body), name)
                except:
                    continue

            # -- Split is not reached yet -- #
            else: # --> Somewhere here will be a split
                try:
                    # -- If the module is already in body we do not need to add it a second time -- #
                    _ = attrgetter('.'.join([*parent, name]))(body)
                except:
                    # -- Add this whole module with all children to body since it does not exist yet -- #
                    body.add_module(name, module)
                # -- Only go into recursion when the name of the layer is equal to where the split is desired -- #
                if name == self.split[len(parent)]:
                    # -- Set the parent node name -- #
                    parent.append(name)
                    # -- Continue with children of current model --> use module -- #
                    body, head, layer_id = self._split_model_recursively_into_body_head(layer_id+1, module, body, head, parent)
        
        # -- Once it is finished, return body and head -- #
        return body, head, layer_id

    def _assemble_model(self, task):
        r"""This function assembles a Generic U-Net based on self.body and corresponding head based
            provided task. The assembled model can than be used with self.model. 
        """
        # -- Check if this task is not already activated -- #
        if self.active_task == task and not isinstance(self.model, Generic_UNet):   # Never want a GenericUNet as self.model --> problems with restoring
            return self.model # --> Nothing to do, task already set

        # -- Assert if the task does not exist in the ModuleDict -- #
        assert task in self.heads.keys(),\
            "The provided task \'{}\' is not a known head, so either initialize the task or provide one that already exists: {}.".format(task, list(self.heads.keys()))

        # -- Extract the corresponding head based on the task -- #
        head = self.heads[task]

        # -- Assemble the model based on self.body and head -- #
        # -- If self.model should be DataParallel, than make it so -- #
        if self.data_parallel: # Define model as nn.DataParallel
            assembled_model = nn.DataParallel(nn.Module())  # --> Do not forget to drop this module later on ..
        else: # Define model as simple nn.Module
            assembled_model = nn.Module()
        
        # -- Add the full body to assembled_model with respect to its name -- #
        for name, module in self.body.named_children():
            assembled_model.add_module(name, module)

        # -- Loop through corresponding head and add the modulew based on their names to assembled_model -- #
        layer_id = len(self.split)-1
        for name, head_part in head.named_children():
            # -- At this point no module is split in between itself anymore, so there is no fusion, only concatenation -- #
            # -- When we are at the top layer, add the head_part directly, no extraction necessary -- #
            if layer_id == 0:
                assembled_model.add_module(name, head_part)
            else:
                # -- Add the head_part based on the current layer_id -- #
                attrgetter('.'.join(self.split[:layer_id]))(assembled_model).add_module(name, head_part) 
                # -- Reduce layer_id so the following modules will be added to the right level -- #
                layer_id -= 1

        # -- Set the active_task -- #
        self.active_task = task

        # -- If self.model should be DataParallel, then remove the 'module' attribute that is an empty module -- #
        if self.data_parallel:
            # -- The empty module used for initialization has always the name 'module' since it can not be changed -- #
            delattr(assembled_model, 'module')    # --> Remove it

        # -- Get all attributes (except the ones that start with '__') from the model --> set the same in assembled_model -- #
        attributes = [a for a in dir(self.model) if not a.startswith('__')]
        
        # -- Loop through these attributes and set assembled_model accordingly -- #
        for attribute in attributes:
            # -- Set assembled_model accordingly if the attributes do not exist -- #
            # -- NOTE: Since self.model is a Generic_UNet from the beginning on -- #
            # --       the functions from Generic_UNet still work --> just the attributes -- #
            # --       were copied, the functions will use self... and thus the right attributes :) -- #
            try: # Try to access the attribute
                getattr(assembled_model, attribute)
            except: # At this point, the attribute does not exist, so set it
                setattr(assembled_model, attribute, getattr(self.model, attribute))

        # -- Set self.model as assembled model -- #
        self.model = assembled_model

        # -- Return the assembled model -- #
        return assembled_model

    # -- Getter and Setters -- #
    def _get_heads(self):
        r"""This function returns the ModuleDict with the multiple heads.
        """
        # -- Return the ModuleDict -- #
        return self.heads

    def _get_body(self):
        r"""This function returns the Module representing the body of the model. 
        """
        # -- Return the Module representing the body for all tasks -- #
        return self.body

    def _set_heads(self, heads):
        r"""This function updates the ModuleDict with the multiple heads.
        """
        # -- Check that heads are of desired instance -- #
        assert isinstance(heads, nn.ModuleDict), "Provided heads are not a nn.ModuleDict.."

        # -- Update the heads -- #
        self.heads = heads

    def _set_body(self, body):
        r"""This function updates the Module representing the body of the model. 
        """
        # -- Check that body is of desired instance -- #
        assert isinstance(body, nn.Module), "Provided body is not a nn.Module.."

        # -- Update the body -- #
        self.body = body