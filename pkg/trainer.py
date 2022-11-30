import os
import pickle
import numpy as np
import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .models.util import to_onehot

class Hyper:
    steps_log = 10
    steps_eval = 50
    steps_save = 10000
    save_model_only = True

    # optimizer
    max_lr = 0.001
    base_lr = 0.001
    lr_decay_factor = 0.99
    lr_patience = 300
    scheduler_mode = 'triangular2'
    step_size_up = 10
    
    ema = 0.99
    max_grad_norm = 10.0
    
    adam_alpha = 2e-4
    adam_betas = (0.5, 0.9)
    adam_eps = 1e-6
    weight_decay = 0.0


    


class Trainer:
    def __init__(self, models, ds_trains, ds_evals=None, hp=Hyper, log_dir=None,
                 optimizers=None, schedulers=None, global_step=0): 
        def _make_dict(objects):
            if objects is None:
                return {}
            elif isinstance(objects, dict):
                return {k:obj for k, obj in objects.items()}
            elif isinstance(objects, list) or isinstance(objects, tuple):
                return {k:obj for k, obj in enumerate(objects)}
            else:
                return {0: objects}
            
        self.models = _make_dict(models)
        self.ds_trains = _make_dict(ds_trains)
        self.ds_evals = _make_dict(ds_evals)
        
        if optimizers is None:
            _params = []
            for m in self.models.values():
                _params.extend(list(m.parameters()))            
            optimizers = torch.optim.SGD(_params, lr=hp.max_lr, weight_decay=hp.weight_decay)
            optimizers = torch.optim.Adam(_params, lr=hp.initial_lr, betas=hp.adam_betas, eps=hp.adam_eps, weight_decay=hp.weight_decay)
        self.optimizers = _make_dict(optimizers)
        
#        if schedulers is None:
#            schedulers = {k: torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.base_lr, max_lr=hp.max_lr, 
#                             step_size_up=hp.step_size_up, mode=hp.scheduler_mode) for k, optimizer in self.optimizers.items()}
#        self.schedulers = _make_dict(schedulers)
        
        self.hp = hp
        self.global_step = global_step
        
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.logger = TBLogger(log_dir) if log_dir else None    
        
        self.reset_data_loaders(train=True)
        self.reset_data_loaders(train=False)

        self.write_config()
        
        
        
               
    def reset_data_loaders(self, train=True, key=0):
        hp = self.hp
        if train and not hasattr(self, 'train_loaders'):
            self.train_loaders = {}
            for k in self.ds_trains.keys():
                self.reset_data_loaders(train, k)
            return
        if not train and not hasattr(self, 'eval_loaders'):
            self.eval_loaders = {}
            for k in self.ds_evals.keys():
                self.reset_data_loaders(train, k)
            return
            
        if hp.batch_type == 'normal':
            if train:
                self.train_loaders[key] = iter(self.ds_trains[key].get_loader(hp.batch_size, num_workers=hp.num_workers))
            else:
                self.eval_loaders[key] = iter(self.ds_evals[key].get_loader(hp.batch_size, num_workers=1))         
        elif hp.batch_type == 'length_sorted':
            if train:
                self.train_loaders[key] = iter(self.ds_trains[key].get_length_sorted_loader(hp.batch_size, num_workers=hp.num_workers))
            else:
                self.eval_loaders[key] = iter(self.ds_evals[key].get_length_sorted_loader(hp.batch_size, num_workers=1))
        else:
            if train:
                self.train_loaders[key] = iter(self.ds_trains[key].get_batch_length_loader(hp.batch_total_length, 
                                                                               num_workers=hp.num_workers))
            else:    
                self.eval_loaders[key] = iter(self.ds_evals[key].get_batch_length_loader(hp.batch_total_length, 
                                                                             num_workers=1))
      
        
        
    def get_batch(self, train=True, key=0, device='cpu'):
        try:
            if train:
                batch = next(self.train_loaders[key])
            else:
                batch = next(self.eval_loaders[key])
        except StopIteration as ex:
            self.reset_data_loaders(train, key)
            if train:
                batch = next(self.train_loaders[key])
            else:
                batch = next(self.eval_loaders[key])
        finally:
            batch = {k: data.to(device) for k, data in batch.items()}
            return batch   
    
    
    def get_loss(self, batch, loss_key=None, backward=False) -> (dict, dict, dict): # losses, outputs, images
        raise NotImplementedError    
        
    def update(self, optimizer, max_grad_norm=None, zero_grad=True):
        if max_grad_norm is not None or zero_grad:
            params = []
            for i, pg in enumerate(optimizer.param_groups):
                params.extend(pg['params'])
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        if zero_grad:
            optimizer.zero_grad()
        
    def train_step(self, **kwargs):
        model = self.models[0]
        model.train()
        device=list(model.parameters())[0].device
        batch = self.get_batch(device=device)
        losses, outputs, images = self.get_loss(batch, backward=True)
        self.update(self.optimizers[0], self.hp.max_grad_norm, zero_grad=True)
        self.global_step += 1
        return losses, outputs, images
    
    def evaluate(self, **kwargs):
        model = self.models[0]
        model.eval()
        device=list(model.parameters())[0].device
        losses, outputs, images = {}, {}, {}
        for key in self.ds_evals.keys():
            _batch = self.get_batch(train=False, key=key, device=device)
            _losses, _outputs, _images = self.get_loss(_batch, backward=False)
            for k, v in _losses.items():
                losses[str(k) + '/' + str(key)] = v 
            for k, v in _outputs.items():
                outputs[str(k) + '/' + str(key)] = v 
            for k, v in _images.items():
                images[str(k) + '/' + str(key)] = v 
        return losses, outputs, images
               
    def schedule_lr(self, key=None, **kwargs):
        if key is None:
            for sch in self.schedulers.values():
                sch.step()
        else:
            self.schedulers[key].step()  
            
             
    def train(self, max_steps=100000000, detect_anomaly=False):
        hp = self.hp
        self.save(model_only=hp.save_model_only)  
                
        for i in range(max_steps):
            _start = time.time()
            if detect_anomaly:
                with torch.autograd.detect_anomaly():
                    losses, outputs, images = self.train_step()
            else:
                losses, outputs, images = self.train_step()
            elapsed = time.time() - _start
                
            if self.global_step % hp.steps_log == 0 and self.logger is not None:
                self.write_log(losses, images=None, train=True)
                print('step:', self.global_step, 'train:', elapsed, 'elapsed, loss:', losses['total'].detach().cpu().numpy())
                
            if self.global_step % hp.steps_eval == 0:
                _start = time.time()
                _losses, _outputs, _images = self.evaluate()
                _elapsed = time.time() - _start
                self.write_log(_losses, _images, train=False)
                lr = list(self.optimizers.values())[0].state_dict()['param_groups'][0]['lr']
                self.logger.scalar_summary('_lr', lr, self.global_step)


            if self.global_step % hp.steps_save == 0:
                self.save(model_only=hp.save_model_only)                
                
            
    def write_log(self, scalars=None, images=None, train=True, tag=None):
        prefix = 'train_' if train else 'eval_'
        if tag is not None:
            prefix += tag + '_'
        if scalars is not None:
            for key, value in scalars.items():
                self.logger.scalar_summary(prefix + key, value, self.global_step)
        if images is not None:
            for key, value in images.items():
                if isinstance(value, list) or isinstance(value, tuple):
                    image, mask0 = value
                else:
                    image, mask0 = value, None
                if image.dim() == 3:
                    image = image[0]
                if mask0 is not None and mask0.dim() == 3:
                    mask0 = mask0[0]                    
                self.logger.image_summary(prefix + key, image, self.global_step, mask0=mask0)        
        

    def save(self, model_only=True):
        save_path = os.path.join(self.log_dir, 'trained_{}.pth'.format(self.global_step))
        state_dict = {}
        state_dict['global_step'] = self.global_step
        model_dict = {}
        for k, m in self.models.items():
            model_dict[k] = m.state_dict()
        state_dict['models'] = model_dict
        if not model_only:
            optim_dict = {}
            for k, optim in self.optimizers.items():
                optim_dict[k] = optim.state_dict()
            state_dict['optimizers'] = optim_dict
        torch.save(state_dict, save_path)
        
    
    def load(self, save_path):
        state_dict = torch.load(save_path, map_location='cpu')
        if 'models' in state_dict.keys(): 
            model_dict = state_dict['models']
            for k, sd in model_dict.items():
                self.models[k].load_state_dict(sd)
        if 'global_step' in state_dict.keys():
            self.global_step = state_dict['global_step']
        if 'optimizers' in state_dict.keys():
            optim_dict = state_dict['optimizers']
            for k, sd in optim_dict.items():
                self.optimizers[k].load_state_dict(sd)
    
    
    def write_config(self):            
        with open(os.path.join(self.log_dir, 'config.txt'), 'w') as f:
            print('---------------------- hp -------------------\n')
            f.write('---------------------- hp -------------------\n')
            for k, v in self.hp.__dict__.items():
                print(k, str(v))
                f.write(k + ' : ' + str(v) + '\n')            
                
            print('--------------------- model cfg -------------------\n')
            f.write('--------------------- model cfg -------------------\n')
            for k, model in self.models.items():
                print('************** model', k, '****************\n')
                print(model.current_config())
                f.write('************** model' + str(k) + '****************\n')
                f.write(str(model.current_config()))

                
        path_cfg = os.path.join(self.log_dir, 'cfg.pkl')
        with open(path_cfg, 'wb') as f:
            cfgs = {k: model.current_config() for k, model in self.models.items()}
            pickle.dump(cfgs, f)
            
        path_hp = os.path.join(self.log_dir, 'hp.pkl')
        with open(path_hp, 'wb') as f:
            pickle.dump(self.hp, f)
            
            
            
            

            
            

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
from PIL import Image

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        
        tag = tag.replace('.', '/')
        if type(value) == torch.Tensor:
            value = value.detach().cpu().numpy()
        try:
            value = np.asscalar(value)
        except Exception:
            pass               
        
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


    def image_summary(self, tag, image, step, mask0=None):
        """Log a list of images."""
        tag = tag.replace('.', '/')
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if mask0 is not None:
            if isinstance(mask0, torch.Tensor):
                mask0 = mask0.detach().cpu().numpy()
            mask1 = 1.0 - mask0
        
        if mask0 is None:
            _max = image.max()
            _min = image.min()
            _range = _max - _min + 1e-10
            image = (image - _min) / _range
        else:
            _max = (image - 1e20 * mask1).max()
            _min = (image + 1e20 * mask1).min()
            _range = _max - _min + 1e-10
            image = (image - _min) * mask0 / _range
         
    
    
        img_summaries = []
        # Write the image to a string
        try:
            s = StringIO()
        except:
            s = BytesIO()
#        scipy.misc.toimage(image).save(s, format="png")
        Image.fromarray((image * 255).astype(np.uint8)).save(s, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=image.shape[0],
                                       width=image.shape[1])
        # Create a Summary value
        img_summaries.append(tf.Summary.Value(tag=tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        tag = tag.replace('.', '/')
        if type(values) == torch.Tensor:
            values = values.detach().cpu().numpy()
            
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

