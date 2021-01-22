import time
import numpy as np

def modelsize(model,type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar='[{}] {}/{} ['.format(self.desc,current,self.n_total)
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = "\r{}".format(bar)
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = ' - ETA: {}'.format(eta_format)
        else:
            if time_per_unit >= 1:
                time_info = ' {:.3f}s/step'.format(time_per_unit)
            elif time_per_unit >= 1e-3:
                time_info = ' {:.3f}ms/step'.format(time_per_unit*1e3)
            else:
                time_info = ' {:.3f}us/step'.format(time_per_unit*1e6)

        show_bar += time_info
        if len(info) != 0:
            show_info = '{} '.format(show_bar) + "-".join([' {}: {:.6f} '.format(key,value) for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
