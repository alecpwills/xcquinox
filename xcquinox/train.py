import equinox as eqx
import optax, sys, gc, jax, os
from jax.interpreters import xla

class xcTrainer(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    loss: eqx.Module
    steps: int
    print_every: int
    clear_every: int
    memory_profile: bool
    verbose: bool
    do_jit: bool
    opt_state: tuple
    serialize_every: int
    logfile: str
    loss_v: float
    
    def __init__(self, model, optim, loss, steps=50, print_every=1, clear_every=1, memory_profile=False, verbose=False, do_jit=True, serialize_every = 1, logfile=''):
        '''
        The base xcTrainer class, whose forward pass computes the training loop.

        :param model: The network which will be trained
        :type model: xcquinox.xc.eXC
        :param optim: optax optimizer object, e.g. optax.adamw(1e-4)
        :type optim: optax.GradientTransformation
        :param loss: The loss function object that computes the loss to be trained against.
        :type loss: eqx.Module
        :param steps: Length of the training cycle, i.e. the number of epochs, defaults to 50
        :type steps: int, optional
        :param print_every: The number of epochs between loss information printing, defaults to 1
        :type print_every: int, optional
        :param clear_every: The number of epochs between calls to clear the cache, defaults to 1
        :type clear_every: int, optional
        :param memory_profile: If True, will write memory profiles at every epoch, to be used with `pprof`, defaults to False
        :type memory_profile: bool, optional
        :param verbose: If true, will print various extra information during the training cycle, defaults to False
        :type verbose: bool, optional
        :param do_jit: Controls whether the update function is jitted or not, useful for debugging if False, defaults to True
        :type do_jit: bool, optional
        :param serialize_every: Controls how often the checkpoint network is written to disk, defaults to 1
        :type serialize_every: int, optional
        '''
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss = loss
        self.steps = steps
        self.print_every = print_every
        self.clear_every = clear_every
        self.memory_profile = memory_profile
        self.verbose = verbose
        self.do_jit = do_jit
        self.serialize_every = serialize_every
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.logfile = logfile
        self.loss_v = 0

    def __call__(self, epoch_batch_len, model, *loss_input_lists):
        '''
        Forward pass of the xcTrainer object, which goes through the training cycle.

        *loss_input_lists are positional arguments, each a list of length [epoch_batch_len elements], corresponding to the proper input order and values for the self.loss function
        I.e., for the E_loss object, these would be [density matrix list], [reference energy list], [ao_eval list], [grid_weight list]

        .. todo: Remove model from inputs, not used since retrieved from self at first iteration
        
        :param epoch_batch_len: The number of batches in a given epoch (i.e., the number of molecules one is training on that are looped over)
        :type epoch_batch_len: int
        :param model: The baseline model to update in the training process
        :type model: xcquinox.xc.eXC
        :return: The updated model after the training cycle completes
        :rtype: xcquinox.xc.eXC
        '''
        BEST_LOSS = 1e10
        for step in range(self.steps):
            jax.debug.print('Epoch {}'.format(step))
            epoch_loss = 0
            if step == 0 and self.logfile:
                with open(self.logfile+'.dat', 'w') as f:
                    f.write('#Epoch\tLoss\tBest\n')
                with open(self.logfile+'batch.dat', 'w') as f:
                    f.write('#Epoch\tBatch\tLoss\tBest\n')
            if step == 0 and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            elif ( (step % self.clear_every) == 0 ) and (step > 0) and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            else:
                fmake_step = self.make_step
            if step == 0:
                print('Step = 0: initializing inp_model and inp_opt_state.')
                inp_model = self.model
                start_model = self.model
                inp_opt_state = self.opt_state
            for idx in range(epoch_batch_len):  
                jax.debug.print('Epoch {} :: Batch {}/{}'.format(step, idx+1, epoch_batch_len))

                #loops over every iterable in loss_input_lists, selecting one batch's input data
                #assumes separate lists, each having inputs for multiple cases in the training set
                loss_inputs = [inp[idx] for inp in loss_input_lists]
                
                # this_loss = self.loss(inp_model, *loss_inputs).item()                
                inp_model, inp_opt_state, this_loss = fmake_step(inp_model, inp_opt_state, *loss_inputs) 
                
                if self.memory_profile:
                    this_loss.block_until_ready()
                    jax.profiler.save_device_memory_profile(f"memory{step}_{idx}.prof")
    
                jax.debug.print('Batch Loss = {}'.format(this_loss))
                if self.logfile:
                    with open(self.logfile+'batch.dat', 'a') as f:
                        f.write(f'{step}\t{idx}\t{this_loss}\t{BEST_LOSS}\n')
                epoch_loss += this_loss
                if ( (step % self.clear_every) == 0 ) and ( (step > 0) == 0 ):
                    eqx.clear_caches()
                    jax.clear_backends()
                    jax.clear_caches()
            #update self.loss_v to epoch's loss
            object.__setattr__(self, 'loss_v', epoch_loss.item())

            if ( (step % self.serialize_every) == 0) and (epoch_loss.item() < BEST_LOSS):
                eqx.tree_serialise_leaves('xc.eqx.{}'.format(step), start_model)
                BEST_LOSS = epoch_loss.item()
            
            #this will persist until next pass
            #the inp_model output just now is fed into get the loss next time, so if better we want to save this
            #not the updated one.
            start_model = inp_model

            if ( (step % self.print_every) == 0 ) or (step == self.steps - 1):
                jax.debug.print(
                    f"{step}, epoch_train_loss={epoch_loss}"
                )
            if ( (step % self.clear_every) == 0 ) and (step > 0):
                eqx.clear_caches()
                jax.clear_backends()
                jax.clear_caches()

            if self.logfile:
                with open(self.logfile+'.dat', 'a') as f:
                    f.write(f'{step}\t{epoch_loss}\t{BEST_LOSS}\n')

        return inp_model

    def make_step(self, model, opt_state, *args):
        '''
        The update step for the training cycle.

        *args input are the inputs to the self.loss function.

        :param model: The model whose weights and biases are to be updated given the loss in self.loss
        :type model: xcquinox.xc.eXC
        :param opt_state: The state of the optimizer to drive the update
        :type opt_state: result of optim.update
        :return: The updated model
        :rtype: xcquinox.xc.eXC
        '''
        self.vprint('loss_value, grads')
        loss_value, grads = eqx.filter_value_and_grad(self.loss)(model, *args)
        self.vprint('updates, opt_state')
        updates, opt_state = self.optim.update(grads, opt_state, model)
        self.vprint('model update')
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # def __post_init__(self, attr, value):
    #     object.__setattr__(self, attr, value)
    
    def clear_caches(self):
        '''
        A function that attempts to clear memory associated to jax caching
        '''
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                if module_name not in ["jax.interpreters.partial_eval"]:
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if hasattr(obj, "cache_clear"):
                            try:
                                obj.cache_clear()
                            except:
                                pass
        gc.collect()


    def vprint(self, output):
        '''
        Custom print function. If self.verbose, will print the called output.

        :param output: The string or value to be printed.
        :type output: printable object
        '''
        if self.verbose:
            jax.debug.print(output)

