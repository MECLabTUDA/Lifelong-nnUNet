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
                         have a name, this specifies it.
            :param prev_trainer: If the split should be performed on previously trained model, than this can be provided
                                 with this variable. Note: The type should be equal to the class_object.
            :param *args, **kwargs: Provide all further positional and keyword arguments that are necessary by the class_object
                                    class when performing an initialization. Note: This needs to be done correctly, since
                                    if it is not, class_object has missing/too much positional arguemnts and will fail during
                                    runtime in initialization.
            NOTE: The model that can be accessed using self.model represents the running model, and is of the same type as 
                  class_object. self btw. is a MultiHead_Module consisting of a nody (self.body), heads (self.heads) and
                  the running model (self.model) based on the activated task (self.active_task). When training on a new task
                  is desired, the useer needs to actiavte this task before calling .forward() in such a way that the running
                  model has the right parameters and structure loaded etc. Tasks are actiavted using self.assemble_model(..).
                  As mentione before, prev_trainer is not necessary but can be used when a pre-trained model should be used
                  as initialization and not a complete new initialized class_object. Further, all splits that will be added
                  using self.add_new_task(..) use the Module that is extracted due to the split from the prev_trainer model,
                  including its state_dict etc.
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
        self.body, self.init_module, _, _ = self._split_model_recursively_into_body_head(layer_id=0, model=self.model) # Start from root with full model
        
        # -- Register the init_module into self.heads based on self.active_task
        self.add_new_task(self.active_task)

        # -- Assemble the model so it can be used for training -- #
        self.assemble_model(task)
        #print(self.replace_layers(self.heads[task], nn.Conv2d, nn.AdaptiveMaxPool2d((6, 6)))) 

    def forward(self, x):
        r"""Forward pass during training --> task needs to be specified before calling forward.
            Assemble the model based on self.body and selected task. Then use parent class to perform
            forward pass. Follow with a split of the model after the pass to update the head and the body.
        """
        # -- Let the class_object do the work since the assembled model is an object of this class -- #
        res = self.class_object.forward(self.model, x) # --> Do not use super, since we want to set the correct self object ;)

        # -- Update the body and corresponding head so these variables are always up to date -- #
        # -- Simply do a splitting again -- #
        self.body, self.heads[self.active_task], _, _ = self._split_model_recursively_into_body_head(layer_id=0, model=self.model)

        # -- Return the forward result generated by Generic_UNet.forward -- #
        return res
    
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
                body.add_module(name, module)

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
                        #head.add_module(head_part[0][0], head_part[0][1])
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
                        # -- due to loop all aunts, uncles, etc. will be added as well after that -- #
                        # -- Loop through head_part and add them to the head -- #
                        for name_part, part in head_part:
                            if len(parent) > 0: # Still in a deeper layer
                                setattr(attrgetter('.'.join(parent))(head), name, module)
                            else:   # No parent exist, so we are on first layer, ie. depth 0
                                setattr(head, name, module)
                            
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
        
        # -- Once it is finished, return body and head -- #
        return body, head, layer_id, parent[:-1]

    def assemble_model(self, task):
        r"""This function assembles a desired model structure based on self.body and corresponding head based
            on the provided task. The parameters of the self.model are updated using the assembled_model composing
            of self.body and specified head.
        """
        # -- Check if this task is not already activated -- #
        if self.active_task == task:
            return self.model # --> Nothing to do, task already set

        # -- Assert if the task does not exist in the ModuleDict -- #
        assert task in self.heads.keys(),\
            "The provided task \'{}\' is not a known head, so either initialize the task or provide one that already exists: {}.".format(task, list(self.heads.keys()))

        # -- Extract the corresponding head based on the task -- #
        head = self.heads[task]

        # -- Assemble the model based on self.body and head to update self.model afterwards -- #
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

        # -- Update the assembled_model attributes -- #
        self._update_model_modules(assembled_model)

        # -- Return the updated model since it might be used in a calling function (inheritance) -- #
        return self.model

    def _update_model_modules(self, model):
        r"""Monkey-patch the self.model in such a way, that the state dicts are correct
            and the functions do properly work using the provided model.
        """
        # -- Loop through the provided model and set (monkey-path) the module in the GenericUNet -- #
        for name, module in model.named_children():
            setattr(self.model, name, module)
    
    def add_n_tasks_and_activate(self, list_of_tasks, activate_with):
        r"""Use this function to initialize for each task name in the list a new head. --> Important when restoring a model,
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
        self.assemble_model(activate_with)

    # -- Getter and Setters -- #
    def get_heads(self):
        r"""This function returns the ModuleDict with the multiple heads.
        """
        # -- Return the ModuleDict -- #
        return copy.deepcopy(self.heads)

    def get_body(self):
        r"""This function returns the Module representing the body of the model. 
        """
        # -- Return the Module representing the body for all tasks -- #
        return copy.deepcopy(self.body)

    def set_heads(self, heads):
        r"""This function updates the ModuleDict with the multiple heads.
        """
        # -- Check that heads are of desired instance -- #
        assert isinstance(heads, nn.ModuleDict), "Provided heads are not a nn.ModuleDict.."

        # -- Update the heads -- #
        self.heads = heads

    def set_body(self, body):
        r"""This function updates the Module representing the body of the model. 
        """
        # -- Check that body is of desired instance -- #
        assert isinstance(body, nn.Module), "Provided body is not a nn.Module.."

        # -- Update the body -- #
        self.body = body

    def get_model_type(self):
        r"""Simply return the assembled model object type which is the same
            as the class object name.
        """
        # -- Return the class objects name -- #
        return self.model.__class__.__name__

    def get_split_path(self):
        r"""This function returns the path to the layer, where the split has been performed.
        """
        # -- Return the split path -- #
        return '.'.join(self.split)

    def add_new_task(self, task):
        r"""Use this function to add the initial module from on the first split.
            Specify the task name with which it will be registered in the ModuleDict.
            NOTE: If the task already exists, it will be overwritten.
        """
        # -- Create a new task in self.heads with the module from the first split -- #
        self.heads[task] = copy.deepcopy(self.init_module)

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