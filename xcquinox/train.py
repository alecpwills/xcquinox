import equinox as eqx
import optax, sys, gc, jax
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
    
    def __init__(self, model, optim, loss, steps=50, print_every=1, clear_every=1, memory_profile=False, verbose=False, do_jit=True):
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
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array)) 
    
    # def __post_init__(self, attr, value):
    #     object.__setattr__(self, attr, value)
    
    def clear_caches(self):
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
        if self.verbose:
            print(output)

    def make_step(self, model, opt_state, *args):
        self.vprint('loss_value, grads')
        loss_value, grads = eqx.filter_value_and_grad(self.loss)(model, *args)
        self.vprint('updates, opt_state')
        updates, opt_state = self.optim.update(grads, opt_state, model)
        self.vprint('model update')
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value


    def __call__(self, epoch_batch_len, model, *loss_input_lists):
        
        for step in range(self.steps):
            print('Epoch {}'.format(step))
            epoch_loss = 0
            if step == 0 and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            elif (step % self.clear_every) == 0 and (step > 0) and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            else:
                fmake_step = self.make_step
            if step == 0:
                inp_model = self.model
                inp_opt_state = self.opt_state
            for idx in range(epoch_batch_len):  
                print('Epoch {} :: Batch {}/{}'.format(step, idx, epoch_batch_len))

                #loops over every iterable in loss_input_lists, selecting one batch's input data
                #assumes separate lists, each having inputs for multiple cases in the training set
                loss_inputs = [inp[idx] for inp in loss_input_lists]
                
                this_loss = self.loss(inp_model, *loss_inputs).item()                
                inp_model, inp_opt_state, train_loss = fmake_step(inp_model, inp_opt_state, *loss_inputs) 

                if self.memory_profile:
                    this_loss.block_until_ready()
                    jax.profiler.save_device_memory_profile(f"memory{step}_{idx}.prof")
    
                print('Batch Loss = {}'.format(this_loss))
                epoch_loss += this_loss
                if (step % self.clear_every) and (step > 0) == 0:
                    fmake_step._clear_cache()
                    eqx.clear_caches()
                    jax.clear_backends()
                    jax.clear_caches()
                    self.clear_caches()
                    self.loss.clear_cache()
                    xla._xla_callable.cache_clear()
    
            if (step % self.print_every) == 0 or (step == self.steps - 1):
                print(
                    f"{step=}, epoch_train_loss={epoch_loss}"
                )
            if (step % self.clear_every) and (step > 0):
                fmake_step._clear_cache()
                eqx.clear_caches()
                jax.clear_backends()
                jax.clear_caches()
                self.clear_caches()
                self.loss.clear_cache()
                xla._xla_callable.cache_clear()

        return model