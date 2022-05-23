#########################################################################################################
#----------This class represents a Generic Module enabling multiple heads for any network.--------------#
#########################################################################################################

import copy
from torch import nn
from typing import Type
from operator import attrgetter

class MultiHead_Module(nn.Module):
    r"""This class is a Module that can be used for any task where multiple heads using a shared body
        are necessary. The heads are stored in a ModuleDict, whereas the task name is the key to the
        corresponding head/module. This class can be used for any network, since the network class object
        needs to be provided as well.
    """
    def __init__(self, class_object: Type[nn.Module], split_at, task, prev_trainer=None, *args, **kwargs):
        r"""Constructor of the Module for multiple heads. 
            :param class_object: This is the class (network) that should be used, ie. that will be split. This needs to
                                 be a class inheriting nn.Module, where the .forward() method is implemented, since this
                                 will be used as well.
            :param split_at: The path in the network where the split should be performed. Use the dot notation ('.') to
                             specify the path to a desired split point.
            :param task: The name of the first head. Since this function will perform a split, the splitted part needs to
                         have a name; this specifies it.
            :param prev_trainer: If the split should be performed on a previously trained/existing model, than this can be provided
                                 with this variable. NOTE: The type needs be equal to the provided class_object.
            :param *args, **kwargs: Provide all further positional and keyword arguments that are necessary by the class_object
                                    class when performing an initialization. NOTE: This needs to be done correctly, since
                                    if it is not, class_object has missing/too much positional arguemnts and will fail during
                                    runtime in initialization. This is only necessary when prev_trainer is not provided or None.
            NOTE: The model that can be accessed using self.model represents the running model, and is of the same type as 
                  class_object. 'self' btw is a MultiHead_Module consisting of a body (self.body), heads (self.heads) and
                  the running model (self.model) based on the activated task (self.active_task). When training on a new task
                  is desired, the useer needs to activate this task before calling .forward() in such a way that the running
                  model has the right parameters and structure loaded etc. Tasks are activated using self.assemble_model(..).
                  As mentioned before, prev_trainer is not necessary but can be used when a pre-trained model should be used
                  as initialization and not a complete new initialized class_object. Further, all tasks that will be added
                  using self.add_new_task(..) use the Module that is extracted based on to the split from the prev_trainer model,
                  including its state_dict etc. if a prev_trainer is used, otherwise the head from a new initialized class_object
                  is used.
        """
        # -- Initialize using super -- #
        super().__init__()

        # -- Store the class_object -- #
        self.class_object = class_object

        # -- If no model is provided, initialize one -- #
        if prev_trainer is None:
            # -- Initialize a conventional network using the provided class_object -- #
            self.model = self.class_object(*args, **kwargs)
        else:
            # -- Ensure that prev_trainer is of same type as class_object, otherwise this will not work -- #
            assert isinstance(prev_trainer, self.class_object),\
                "This function splits a \'{}\' module class object, but a \'{}\' module is provided.".format(self.class_object.__name__, type(prev_trainer))
            # -- Ensure that the prev_trainer is not empty -- #
            assert len(list(prev_trainer.children())) > 0,\
                "When using a prev_trainer, please ensure that it is not empty or do not specify one."
            # -- Set the models class to prev_trainer -- #
            self.model = prev_trainer
            del prev_trainer
            
        # -- Save the split_at, so it can be used throughout the class -- #
        assert isinstance(split_at, str), "The provided split needs to be a string.."
        self.split = [x.strip() for x in split_at.split('.')] # --> Never know what users input so strip layer names
        
        # -- Test if the provided path even exists -- #
        try:
            # -- Extract the part where the split will be and check if it even exists -- #
            _ = attrgetter('.'.join(self.split))(self.model)
        except:
            assert False, "The provided split path \'{}\' does not exist..".format('.'.join(self.split))

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
        
        # -- Define empty ModuleDict for all the heads -- #
        self.heads = nn.ModuleDict()

        # -- Define a variable that specifies the active_task -- #
        assert isinstance(task, (str, int)), "The provided task needs to be an integer (ID) or string, not {}..".format(type(task))
        self.active_task = task

        # -- Split self.model based on split_at and store the init_module -- #
        # -- NOTE: If prev_trainer was used, init_module is now the splitted part from -- #
        # --       the prev_trainer and not from a new generated class_object, ie. this part -- #
        # --       is always used when calling self.add_new_task --> keep this in mind -- #
        self.body, init_module, _, _ = self._split_model_recursively_into_body_head(layer_id=0, model=self.model) # Start from root with full model

        # -- Copy the state_dict from the init_module -- #
        self.state_init = init_module.state_dict()

        # -- Register the init_module into self.heads based on self.active_task
        self.heads[self.active_task] = init_module

        # -- Remove the init_module since it is only used for initializing the first task -- #
        del init_module

        # -- Define a flag that indicates if the body weights are frozen or not -- #
        self.body_freezed = True

        # -- Assemble the model so it can be used for training -- #
        self.assemble_model(task, freeze_body=False)
        # -- Now the body is not freezed anymore and we made sure of that -- #
        self.body_freezed = False

    def forward(self, x):
        r"""Forward pass during training --> task needs to be specified before calling forward.
            Assemble the model before callinng this function.
            NOTE: Follow with a split of the model after the backward pass to update the head and the body;
                  use self.update_after_iteration(..) for it.
        """
        # -- Let the class_object do the work since the assembled model is an object of this class -- #
        res = self.class_object.forward(self.model, x) # --> Do not use super, since we want to set the correct self object ;)

        # -- Return the forward result generated by Generic_UNet.forward -- #
        return res

    def update_after_iteration(self, model=None, update_body=True):
        r"""This function is used to update the head and body. This should be used after every backward pass
            of the network that is trained on.
            :param model: The model that should be split and with which the parameters for self.body and the
                          current head are updated.
            :param update_body: This boolean flag identifies if the body should be updated as well or only the
                                head.
        """
        # -- If model is None set it to self.model -- #
        if model is None:
            model = self.model

        # -- Split the model and update body with the corresponding head if desired -- #
        if update_body: # Update body and head both
            self.body, head, _, _ = self._split_model_recursively_into_body_head(layer_id=0, model=model) # Start from root with full model
        else:   # Do not update the body, ie. only update the head
            _, head, _, _ = self._split_model_recursively_into_body_head(layer_id=0, model=model) # Start from root with full model

        self.heads[self.active_task] = head
        self.heads[self.active_task].load_state_dict(head.state_dict())

    def _split_model_recursively_into_body_head(self, layer_id, model, body=nn.Module(), head=nn.Module(),
                                                parent=list(), simplify_split=False):
        r"""This function splits a provided model into a body and a single head.
            It returns the body, head, layeron_id and a parent_list on which the split is performed.
            :param layer_id: The current index of the layer. 0 is root, 1 is child of root, etc.
            :param model: The model to be split or a submodule of it
            :param body: Representing the base of the initial model since this is a recursive function
            :param head: Representing the head after the split of the initial model since this is a recursive function
            :param parent: List of strings representing the name of the parent module (path to current node in tree)
            :param simplify_split: Bool whether the split should be simplified. Use this only if the function is called from outside.
                                   When it is called from inside, self.split is already simplified, but if it should change throughout
                                   it should be simplified again, then set this to true (wondering if that's ever the case..).
            :return body, head, layer_id, parent: The splitted model (body and head) with the layer_id where the split is performed
                                                  and the list of parents (should be empty at the end, ie. after recursion is finished)
            NOTE: This function is a recursive function, a split should be initialized with self.model and
                  a layer_id of 0. Further, the returned layer_id is for recursion and might not be of interest
                  to the user since at the end it is equal to the length of self.split. Same goes for the return list of parents.
        """
        # -- If the function is called the first time, perform some checks -- #
        if layer_id == 0:
            # -- Assert if the split is empty -- #
            assert len(self.split) != 0, "The provided split is empty.."

            first_layer = next(model.named_children()) # Extract the first layer in the model
            # -- Assert if the user tries to split at very first layer --> makes no sense no split necessary, use class_object -- #
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
        for idx, (name, n_module) in enumerate(list(model.named_children())):
            # -- Deep copy the module to bypass the runtime error that would occur due to changing of the module -- #
            module = copy.deepcopy(n_module)

            # -- If, given layer_id, a desired module from the path is not reached and layer_id is -- #
            # -- smaller than number of split path elements --> set body -- #
            # -- Ensure that nothing is added twice due to recursion, thus layer_id == 0 -- #
            if layer_id == 0 and layer_id < len(self.split)-1 and name != self.split[layer_id]:
                # -- Add this whole module with all children to body -- #
                # body.add_module(name, module)
                body.add_module(name, n_module)

            # -- If the layer_id is equal to the number of split path elements and equal to the last element in split -- #
            # -- the split point is found --> the last node of the split, so everything after that is head -- #
            elif layer_id == len(self.split)-1:
                if name == self.split[layer_id]:
                    # -- Extract the module in which the split is performed -- #
                    if layer_id == 0:   # Split on first layer, no parent exists --> parent is model
                        parent_mod = model
                    else:   # Deeper layer, parent is set by now and already present in the body
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
                        # -- Define all layers from parent so they can be reached accordingly in the head -- #
                        r_p = []    # --> Running parents to keep track of the path
                        for p in parent:
                            if len(r_p) > 0:
                                # -- Set an empty Module --> does not matter for now, since we replace it afterwards -- #
                                setattr(attrgetter('.'.join(r_p))(head), p, nn.Module())
                            else: # First layer needs to be added first
                                setattr(head, p, nn.Module())
                            r_p.append(p)
                        # -- In this step we are in recursion level, so add all the siblings of the current module, -- #
                        # -- due to the loop, all aunts, uncles, etc. will be added as well after that -- #
                        # -- Loop through head_part and add them to the head -- #
                        for name_part, part in head_part:
                            if len(parent) > 0: # Still in a deeper layer
                                setattr(attrgetter('.'.join(parent))(head), name_part, part)
                            else:   # No parent exist, so we are on first layer, ie. depth 0
                                setattr(head, name_part, part)
                            
                    # -- Increase layer_id so all other modules are added to head and removed from body -- #
                    layer_id += 1
                    continue
                    
                # -- Do nothing else, since the part is already in the body and does not belong to the head -- #

            # -- If layer_id is greater than number of split path elements --> set head -- #
            elif layer_id > len(self.split)-1:
                # -- Add this whole module with all children to head -- #
                # -- Use the parent, otherwise layers might be overwritten and -- #
                # -- the parent is necessary for setting everything right -- #
                if len(parent) > 0: # Still in a deeper layer
                    setattr(attrgetter('.'.join(parent))(head), name, module)
                else:   # No parent exist anymore, so we are on first layer, ie. depth 0
                    setattr(head, name, module)
                
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
                    body, head, layer_id, parent = self._split_model_recursively_into_body_head(layer_id+1, module, body, head, parent)

        return body, copy.deepcopy(head), layer_id, parent[:-1]

    def assemble_model(self, task, freeze_body=False):
        r"""This function assembles a desired model structure based on self.body and corresponding head based
            on the provided task. The parameters of the self.model are updated using the assembled_model composing
            of self.body and specified head.
            :param task: Task name of the head to specify which head should be joined with the body.
            :param freeze_body: Specify if the body weights should be freezed or not.
            :return: Function returns the running model (self.model)
        """
        # -- Check if this task is not already activated -- #
        if self.active_task == task and freeze_body == self.body_freezed:
            return self.model # --> Nothing to do, task already set

        # -- Assert if the task does not exist in the ModuleDict -- #
        assert task in self.heads.keys(),\
            "The provided task \'{}\' is not a known head, so either initialize the task or provide one that already exists: {}.".format(task, list(self.heads.keys()))

        # -- Extract the corresponding head based on the task -- #
        head = copy.deepcopy(self.heads[task])

        # -- Assemble the model based on self.body and head to update self.model afterwards -- #
        assembled_model = nn.Module()
        
        # -- Add the full body to assembled_model with respect to its name -- #
        for name, module in self.body.named_children():
            assembled_model.add_module(name, module)

        # -- Add all modules from the head based on their names to assembled_model -- #
        assembled_model, _ = self._join_body_head_recursively(assembled_model, head, list())

        # -- Set the active_task -- #
        self.active_task = task
        
        # -- Load the state_dict from the assembled model into self.model for updating the running model -- #
        self.model.load_state_dict(assembled_model.state_dict())
        del assembled_model

        # -- Freeze the body if desired and not freezed -- #
        if freeze_body and not self.body_freezed:
            # -- Freeze all body weights -- #
            self._set_requires_grad(False)
            # -- Update the internal flag as well -- #
            self.body_freezed = True
        
        # -- Unfreeze the body if desired and freezed -- #
        if not freeze_body and self.body_freezed:
            # -- Unfreeze all body weights -- #
            self._set_requires_grad(True)
            # -- Update the internal flag as well -- #
            self.body_freezed = False

        # -- Return the updated model since it might be used in a calling function (inheritance) -- #
        return self.model

    def _set_requires_grad(self, requires_grad):
        r"""This function is used to freeze/unfreeeze all layers in the body so when training, the body weights are/ are not
            changed during backpropagation.
            :param model: The model for which the body will be freezed/unfreezed.
            :param requires_grad: Specifies if the gradients are required or not. If set to false, the body will be freezed.
                                  If set to True, the body will not be freezed and the weights are updated during
                                  optimizer.step() again.
            :return: The modified model with the freezed/unfreezed body will be returned.
        """
        # -- Extract the parameter names belonging to the body -- #
        body_parameters = [name for name, _ in self.body.named_parameters()]
        # -- Loop through the parameter names of the model -- #
        for name, param in self.model.named_parameters():
            # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
            if name in body_parameters:
                # -- Set requires_grad accordingly -- #
                param.requires_grad = requires_grad

    def _join_body_head_recursively(self, body, head, parents=list()):
        r"""This function is used to join a body with a head in a recursive manner. This function
            is only used internally when assembling a model given a specific task.
        """
        # -- Loop through the modules in the head -- #
        for name, module in head.named_children():
            # -- If there are still children left, go recursive -- #
            if len(list(module.children())) > 0:
                # -- Try to access the module in the body -- #
                try:
                    # -- If it does exist, than move on as used to -- #
                    _ = attrgetter('.'.join([*parents, name]))(body)
                except:
                    # -- NOTE: The modules from the head do not exist in body so simply set the modules at the right position -- #
                    if len(parents) > 0:
                        # -- We are in a deeper layer so use the tracked path to add the module at correct position -- #
                        setattr(attrgetter('.'.join(parents))(body), name, module)
                    else:
                        # -- No parent exist, ie. first layer, than just add the module -- #
                        setattr(body, name, module)

                # -- Add the current name into parents to keep track of the path -- #
                parents.append(name)

                # -- Go one layer deeper -- #
                body, parents = self._join_body_head_recursively(body, module, parents)

            else:
                if len(parents) > 0:
                    # -- We are in a deeper layer so use the tracked path to add the module at correct position -- #
                    setattr(attrgetter('.'.join(parents))(body), name, module)
                else:
                    # -- No parent exist, ie. first layer, than just add the module -- #
                    setattr(body, name, module)

        # -- Return the updated body, head and parent without the current node -- #
        return body, parents[:-1]
    
    def add_new_task(self, task, use_init, model=None):
        r"""Use this function to add the initial module from on the first split.
            Specify the task name with which it will be registered in the ModuleDict.
            :param task: Task name of the new task (key for the ModuleDict)
            :param model: nn.Module that represent the new task. If this is None, the new head
                          is a copy from the very first splitted head during initialization
            NOTE: If the task already exists, it will be overwritten. If the user provides
                  a model, than ensure that it works with the forward function from the
                  class_object. If this does not map than an error will be thrown later on.
        """
        # -- Create a new task in self.heads with the module from the first split -- #
        if model is None:
            # -- Add the latest task -- #
            self.heads[task] = copy.deepcopy(self.heads[list(self.heads.keys())[-1]])
            
            if use_init:
                # -- Load the state_dict from the very first split -- #
                self.heads[task].load_state_dict(self.state_init)
            
            # -- ELSE: The head has the state_dict from the last trained head --> used for transfer learning -- #

        else:
            # -- Register the provided model -- #
            self.heads[task] = copy.deepcopy(model)

    def add_n_tasks_and_activate(self, list_of_tasks, activate_with, remove_old_tasks=True):
        r"""Use this function to initialize for each task name in the list a new head. --> Important when restoring a model,
            since the heads are not created in this init function and need to be set manually before loading a state_dict
            that includes n heads. Further, self.model will be assembled based on activate_with. In the case of calling
            this function before restoring, the user needs to provide the correct activate_with, ie. the head that was activated
            in times of saving the Multi Head Network. otherwise there will be a mixup that might be difficult to spot (wrong head
            with wrong state_dict eg.).
            :param list_of_tasks: List of strings representing the task names
            :param activate_with: String, for instance a task that is used to assemble self.model
            :param remove_old_tasks: Bool, indicating if all heads that are not mentioned in list_of_tasks should be removed
        """
        # -- Loop through list of tasks -- #
        for task in list_of_tasks:
            # -- Add the task to the head if it does not exist-- #
            if task not in self.heads:
                self.add_new_task(task, use_init=True)

        # -- Remove all heads that are not in the list_of_tasks if desired -- #
        if remove_old_tasks:
            for task in list(self.heads.keys()):
                if task not in list_of_tasks:
                    # -- Remove it from the head -- #
                    del self.heads[task]

        # -- Assemble the model based on activate_with -- #
        self.assemble_model(activate_with)

    # -- Getter and Setters -- #
    def get_heads(self):
        r"""This function returns the ModuleDict with the multiple heads.
            :return: Function returns a deepcopy of self.heads
        """
        # -- Return the ModuleDict -- #
        return copy.deepcopy(self.heads)

    def get_body(self):
        r"""This function returns the Module representing the body of the model.
            :return: Function returns a deepcopy of self.body
        """
        # -- Return the Module representing the body for all tasks -- #
        return copy.deepcopy(self.body)

    def set_heads(self, heads, reset=True):
        r"""This function resets (reset=True) or updates (reset=False) the ModuleDict with the multiple heads.
            :param heads: nn.Module which will update or reset self.heads
            :param reset: Boolean specifying if the heads should be deleted and reset or just updated
        """
        # -- Check that heads are of desired instance -- #
        assert isinstance(heads, nn.ModuleDict), "Provided heads are not a nn.ModuleDict."

        # -- Reset or update the head based on the reset flag -- #
        if reset:
            # -- Reset the heads -- #
            del self.heads
            self.heads = heads
        else:
            # -- Update the heads without clearing it -- #
            self.heads.update(heads)
        
    def set_body(self, body):
        r"""This function updates the Module representing the body of the model. 
            :param body: nn.Module which will update self.body
        """
        # -- Check that body is of desired instance -- #
        assert isinstance(body, nn.Module), "Provided body is not a nn.Module.."

        # -- Update the body -- #
        del self.body
        self.body = copy.deepcopy(body)

    def get_model_type(self):
        r"""Simply return the running models object type which is the same
            as the class object name.
        """
        # -- Return the class objects name -- #
        return self.model.__class__.__name__

    def get_split_path(self):
        r"""This function returns the path to the layer, where the split has been performed.
            NOTE: This split is simplified if the user did not change it after simplification.
        """
        # -- Return the split path -- #
        return '.'.join(self.split)

    def replace_layers(self, model, old, new):
        r"""This function is based on: https://www.kaggle.com/ankursingh12/why-use-setattr-to-replace-pytorch-layers.
            It can be used to replace a desired layer in the provided model with a new Module.
            :param model: nn.Module in which the layer (old) should be replaced (with new)
            :param old: nn.Module representing a specific layer, like nn.Conv2d, nn.ReLU, etc.
            :param new: nn.Module representing a specific layer, like nn.Conv2d, nn.ReLU, etc. which will replace old
            :return: Function returns the updated module.
            NOTE: This function can be used for instance to replace e.g. a desired layer in the body/head. It does
                  not have to be a Module directly from torch.nn, it can also be any Module that inherits nn.Module.
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