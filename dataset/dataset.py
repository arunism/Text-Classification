import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Dataset:
    def __init__(self, file) -> None:
        self._file = file

    def _read_file(self) -> None:
        data_l = list()
        with open(self._file, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                data = line.strip().split(':::')[1:]
                data_l.append(data)
            file.close()
        return data_l
    
    def write_to_file(self) -> None:
        data_l = self._read_file()
        outfile = os.path.join(BASE_DIR, 'data.csv')
        df = pd.DataFrame(data_l, columns=['Title', 'Genre', 'Summary'])
        df[df.columns] = df.apply(lambda x: x.str.strip())
        df.to_csv(outfile, index=False)


if __name__ == '__main__':
    train_data_file = os.path.join(BASE_DIR, 'train_data.txt')
    train_data = Dataset(train_data_file)
    train_data.write_to_file()
