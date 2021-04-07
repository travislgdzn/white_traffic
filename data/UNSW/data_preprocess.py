import csv
import pymongo
import time
import numpy as np


class DataFields:
    """
    首先定义一个字段类存储字段信息，包含数据集下的所有字段信息
    """
    fields = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
              'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
              'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
              'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
              'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
              'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']


def check_test_train_fields():
    """
    该方法是作为一个验证使用，主要是从mongo 数据库中读取数据，
    验证其键值对是否一样，包括排序是否相同，如果不相同则程序报错
    """
    client = pymongo.MongoClient()
    db = client['UNSW']
    table_test = db['test']
    table_train = db['train']
    test_fields = [key for key, value in table_test.find_one().items()]
    train_fields = [key for key, value in table_train.find_one().items()]
    for i in range(len(test_fields)):
        if test_fields[i] == train_fields[i]:
            continue
        raise Exception('not same')
    print('len fields', len(test_fields))
    print('all fields is same')


# check_test_train_fields()


class Csv2mongo(object):
    def __init__(self):
        """
        在该初始化方法中定义基本的数据信息，配置信息，
        读取的数据集的位置，以及数据存储在mongo数据库中的位置，
        首先从csv中读取数据信息，存入　UNSW 数据库的test 和 train 表
        ，然后计算每个字段的最大值最小值，计算出每个数据的归一化结果
        将数据存入test_normal 和　train_normal 表
        """
        self.train_csv_path = './data/UNSW_NB15_training-set.csv'
        self.test_csv_path = './data/UNSW_NB15_testing-set.csv'

        client = pymongo.MongoClient()
        db = client['UNSW_show_1']
        self.table_test = db['test']
        self.table_train = db['train']
        self.table_test_normal = db['test_normal']
        self.table_train_normal = db['train_normal']
        self.table_fields = db['fields_info']
        # print()

    def find_counts(self):
        """
        该函数主要是统计训练集中的数据字段信息和
        　Datafields 类中的字段信息是否一致
        """
        oringin_fields = [i.lower() for i in DataFields.fields]
        _test_fields = [i.lower() for i, value in self.table_train.find_one().items()]
        _test_fields.remove('_id')
        print(oringin_fields)
        print(_test_fields)
        both_have = [i for i in oringin_fields if i in _test_fields]
        oringin_fields_has_test_not = [i for i in oringin_fields if i not in _test_fields]
        _test_fields_has_origin_not = [i for i in _test_fields if i not in oringin_fields]
        print(both_have)
        print(oringin_fields_has_test_not)
        print(_test_fields_has_origin_not)

    def build_fields_map(self, train_path, test_path):
        """
        从csv表中读取所有数据，构建　字段字典，
        使所有字符串类型的数据可以依据该字典信息得到一个合适的唯一的数字码，
        每次运行该程序的时候都会创建
        一张新的字典表，所以每次运行结束之后，每个字符串对应的数字不是不变的
        """
        with open(train_path, 'r') as f:
            lines = csv.reader(f)
            lines = [i for i in lines]
        titles = lines.pop(0)
        titles[0] = 'id'
        with open(test_path, 'r') as f:
            test_lines = csv.reader(f)
            test_lines = [i for i in test_lines]
        test_titles = test_lines.pop(0)
        test_titles[0] = 'id'
        if titles != test_titles:
            raise Exception('titles is not same')
        lines.extend(test_lines)
        data_ins = [dict(zip(titles, line)) for line in lines]
        proto = list(set(tuple([i['proto'] for i in data_ins])))
        service = list(set(tuple([i['service'] for i in data_ins])))
        state = list(set(tuple([i['state'] for i in data_ins])))
        attack_cat = list(set(tuple([i['attack_cat'] for i in data_ins])))
        proto_map = {str(key): value for key, value in enumerate(proto)}
        service_map = {str(key): value for key, value in enumerate(service)}
        state_map = {str(key): value for key, value in enumerate(state)}
        attack_map = {str(key): value for key, value in enumerate(attack_cat)}
        maps = {'proto': proto_map,
                'service': service_map,
                'state': state_map,
                'attack_cat': attack_map}
        self.table_fields.drop()
        self.table_fields.insert_one(maps)

    def csv2mongo(self, path, table):
        """
        该函数就是将csv 的数据转存到mongo 的主要方法
        """
        with open(path, 'r') as f:
            lines = csv.reader(f)
            lines = [i for i in lines]
        titles = lines.pop(0)
        titles[0] = 'id'
        proto_index = titles.index('proto')
        service_index = titles.index('service')
        state_index = titles.index('state')
        attack_cat_index = titles.index('attack_cat')
        hit = self.table_fields.find()[0]
        proto_map = hit['proto']
        service_map = hit['service']
        state_map = hit['state']
        attack_cat_map = hit['attack_cat']
        datas = []
        for line in lines:
            tag = '----this is error'
            line[attack_cat_index] = [key for key, value in attack_cat_map.items() if value == line[attack_cat_index]]
            line[attack_cat_index] = line[attack_cat_index][0] if len(line[attack_cat_index]) == 1 else tag
            line[service_index] = [key for key, value in service_map.items() if value == line[service_index]]
            line[service_index] = (self.one_hot_data(line[service_index][0], len(service_map))
                                   if len(line[service_index]) == 1 else tag)
            line[state_index] = [key for key, value in state_map.items() if value == line[state_index]]
            line[state_index] = (self.one_hot_data(line[state_index][0], len(state_map))
                                 if len(line[state_index]) == 1 else tag)
            line[proto_index] = [key for key, value in proto_map.items() if value == line[proto_index]]
            line[proto_index] = (self.one_hot_data(line[proto_index][0], len(proto_map))
                                 if len(line[proto_index]) == 1 else tag)
            if tag in line:
                print(path, 'error------------------------------')
                print(line)
                continue
            data = dict(zip(titles, line))
            data_in = {}
            for key, value in data.items():
                if isinstance(value, list) and key != 'attack_cat':
                    for i in range(len(value)):
                        data_in.update({'{}_{}'.format(key, i): value[i]})
                else:
                    data_in.update({key: value})
            datas.append(data_in)
        table.drop()
        table.insert_many(datas)

    def one_hot_data(self, index, total):
        """
        该函数构造一个一维数组，再从该一维数组中
        找到一个下标使其返回一个one-hot 编码数据
        """
        data = [0 for i in range(total)]
        data[int(index)] = 1
        return data

    def normal(self):
        """
        该函数将已经存在mongo 的数据取出，
        对其进行数据归一化处理之后将其存入新的表中
        """
        datas = []
        exclude = ['id', 'attack_cat', 'label']
        fields = [key for key, value in self.table_test.find_one().items()]
        fields.remove('_id')
        exclude_indexs = [fields.index(i) for i in exclude]
        print(exclude_indexs, 'excludes fields ---------')
        datas.extend(self.mongo2list(self.table_test, fields))
        datas.extend(self.mongo2list(self.table_train, fields))
        datas = np.array(datas, dtype='float')
        min_list = np.min(datas, axis=0)
        max_list = np.max(datas, axis=0)
        print(min_list, 'min list ---------')
        print(max_list, 'max list -----------')
        max_min_list = max_list - min_list
        print(max_min_list, 'max min list --------------')
        test_datas = np.array(self.mongo2list(self.table_test, fields), dtype='float')
        train_datas = np.array(self.mongo2list(self.table_train, fields), dtype='float')
        test_normal_datas = self.normal_list(test_datas, min_list, max_min_list, exclude_indexs)
        train_normal_datas = self.normal_list(train_datas, min_list, max_min_list, exclude_indexs)
        test_data_ins = [dict(zip(fields, data)) for data in test_normal_datas]
        train_data_ins = [dict(zip(fields, data)) for data in train_normal_datas]
        self.table_test_normal.drop()
        self.table_train_normal.drop()
        self.table_test_normal.insert_many(test_data_ins)
        self.table_train_normal.insert_many(train_data_ins)

    def normal_list(self, datas, min_list, max_min_list, exclude_indexs):
        """
        该函数就是具体计算数据归一化的结果的函数，
        通过传入数据，最小值的列表，最大值的列表，
        需要排除的字段的列表
        返回数据归一化的结果
        """
        normal_datas = []
        for data in datas:
            for i in range(len(max_min_list)):
                if i in exclude_indexs:
                    continue
                if max_min_list[i] > 0:
                    data[i] = (data[i] - min_list[i]) / max_min_list[i]
            normal_datas.append(data)
        return normal_datas

    def mongo2list(self, table, fields):
        """
        该函数按照对应的字段取出值，使得字典中的值按照所给的fields
        的序列组合成一个数组，最后多条数据组合成的数据组合成一个二维数组返回
        """
        datas = []
        for hit in pymongo.cursor.Cursor(table, batch_size=1000):
            hit.pop('_id')
            datas.append([hit[key] for key in fields])
        return datas

    def run(self):
        """
        该类的主函数
        """
        self.build_fields_map(self.train_csv_path, self.test_csv_path)
        self.csv2mongo(self.test_csv_path, self.table_test)
        self.csv2mongo(self.train_csv_path, self.table_train)
        self.normal()
        print('ok-------------------')


if __name__ == '__main__':
    Csv2mongo().run()

    print('----------------')

