from pathlib import Path
from torch import optim

import argparse

project_dir = Path(__file__).resolve().parent
data_dir = project_dir.joinpath('datasets')
data_dict={'tweet': data_dir.joinpath('Tweet')}

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key=='optimizer':
                    value=optimizer_dict[value]
                setattr(self, key, value)

        self.dataset_dir = data_dict[self.data.lower()]
        self.data_dir = self.dataset_dir

        self.data_file_path=self.data_dir.joinpath(self.data_file_name)

    def __str__(self):
        config_str='Configurations\n'
        config_str +=pprint.pformat(self.__dict__)
        return config_str

def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--data', type=str, default='tweet')

    parser.add_argument('--data_file_name', type=str, default='sentiment_analysis.csv')

    parser.add_argument('--model', type=str,
                        default='LSTM', help='one of {LSTM, }')
    parser.add_argument('--n-epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')


    kwargs = parser.parse_args()
    
    kwargs=vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
