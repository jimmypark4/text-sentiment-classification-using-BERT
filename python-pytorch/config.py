from datetime import datetime
from pathlib import Path
from torch import optim
import pprint
import argparse

# ─────────────────────────────────────────────────────────────
# [수정 포인트]
# 원래 'datasets/Tweet' 폴더를 가리키고 있던 부분을
# 'C:/text-sentiment-classification-using-BERT/data'로 직접 지정
# ─────────────────────────────────────────────────────────────

# ※ config.py 파일이 현재
#    C:\text-sentiment-classification-using-BERT\python-pytorch\config.py
#    에 있다고 가정하겠습니다.

# project_dir = Path(__file__).resolve().parent  # ← 기존 코드
# data_dir = project_dir.joinpath('datasets')    # ← 기존 코드
# data_dict = {'tweet': data_dir.joinpath('Tweet')}  # ← 기존 코드

# 아래와 같이 바꿔주면, 'tweet'이라는 데이터를 사용할 때
# C:\text-sentiment-classification-using-BERT\data 경로를 사용하게 됩니다.
data_dir = Path(r'C:\text-sentiment-classification-using-BERT\data')
data_dict = {'tweet': data_dir}

# 옵티마이저 선택 딕셔너리
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

# 설정값을 관리하는 클래스
class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                # 'optimizer' 키 처리
                if key == 'optimizer':
                    value = optimizer_dict[value]
                setattr(self, key, value)

        # ──────────────────────────────────────────────────────────────────
        # [수정 포인트] data_dict에서 self.data.lower() → 'tweet' → data_dir(위에서 지정)
        # ──────────────────────────────────────────────────────────────────
        self.dataset_dir = data_dict[self.data.lower()]
        self.data_dir = self.dataset_dir

        # 실제 CSV 파일(기본: 'sentiment_analysis.csv') 경로 생성
        self.data_file_path = self.data_dir.joinpath(self.data_file_name)

    def __str__(self):
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--data', type=str, default='tweet')  # ← 'tweet' 디폴트
    parser.add_argument('--data_file_name', type=str, default='sentiment_analysis.csv')

    parser.add_argument('--model', type=str, default='LSTMMamba', help='one of {LSTM, }')
    parser.add_argument('--n-epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 시:분:초 대신 시-분-초
    parser.add_argument('--name', type=str, default=f"{time_now}")

    kwargs = parser.parse_args()
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
