import logging
from functools import reduce

import numpy as np
import src.config as config
import src.pg_operating as pg_operating
from bert_serving.client import BertClient
from milvus import *
from pymongo import MongoClient
import pandas as pd


def get_bc():
    bc = BertClient(ip=config.BERT_CLIENT_HOST, check_version=False, check_length=False)
    return bc


index_file_size = 1024
metric_type = MetricType.IP
nlist = 16384

MILVUS_HOST = config.MILVUS_HOST
MILVUS_PORT = config.MILVUS_PORT


# PG_HOST = config.PG_HOST
# PG_PORT = config.PG_PORT
# PG_USER = config.PG_USER
# PG_PASSWORD = config.PG_PASSWORD
# PG_DATABASE = config.PG_DATABASE

def get_mongo():
    conn = MongoClient(config.MONGO_HOST, config.MONGO_PORT)
    db = conn.medicalQa
    my_set = db.open
    return my_set


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def import_to_mongo(ids, answer_file):
    """
    数据导入mongo数据库中
    :param ids:
    :param answer_file:
    :return:
    """
    my_set = get_mongo()
    for i, j in zip(ids, answer_file):
        my_set.insert_one({"ids": i, "answer_file": j})


def normaliz_vec(vec_list):
    """
    标准化向量
    :param vec_list:
    :return:
    """
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        vec_list[i] = vec
    return vec_list


def import_to_milvus(data, collection_name, milvus):
    """
    数据导入milvus中
    :param data:
    :param collection_name:
    :param milvus:
    :return:
    """
    print("开始链接bert")
    bc = get_bc()
    print("开始预测", data)
    vectors = bc.encode(data)
    print("预测结束")
    question_vectors = normaliz_vec(vectors.tolist())
    status, ids = milvus.insert(collection_name=collection_name, records=question_vectors)
    print(status)
    # index_param = {'index_type': IndexType.IVF_SQ8, 'nlist': nlist}
    # status = milvus.create_index(collection_name,index_param)
    # print(status)
    return ids


def create_milvus_table(collection_name, milvus):
    """
    创建milvus表格（单元格）
    :param collection_name:
    :param milvus:
    :return:
    """
    param = {'collection_name': collection_name, 'dimension': 768, 'index_file_size': index_file_size,
             'metric_type': metric_type}
    status = milvus.create_collection(param)
    print(status)
    # index_param = {'index_type': IndexType.IVF_SQ8, 'nlist': nlist}
    # milvus.create_index(collection_name,index_param)


def has_table(collection_name, milvus):
    """
    判断是否存在这个表
    :param collection_name:
    :param milvus:
    :return:
    """
    status, ok = milvus.has_collection(collection_name)
    if not ok:
        # print("create table.")
        create_milvus_table(collection_name, milvus)
        index_param = {'nlist': nlist}
        status = milvus.create_index(collection_name, IndexType.IVF_SQ8, index_param)
        print(status)
    else:
        print(status)
    # print("insert into:", collection_name)


def read_data_txt(file_dir):
    """
    读取txt数据
    :param file_dir:
    :return:
    """
    data = []
    with open(file_dir, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                data.append(line)
    return data


def read_data_csv(file_dir, top_k):
    """
    读取txt数据
    :param file_dir:
    :return:
    """
    data = pd.read_csv(file_dir).drop_duplicates().dropna().values.tolist()[:top_k]
    answer_data = []
    question_data = []
    for i in data:
        answer_data.append(i[0])
        question_data.append(i[1])
    return question_data, answer_data


def import_data(collection_name, question_dir, answer_dir):
    """
    导入数据到milvus中
    :param collection_name:
    :param question_dir:
    :param answer_dir:
    :return:
    """
    question_data = read_data_txt(question_dir)
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    has_table(collection_name, milvus)
    ids = import_to_milvus(question_data, collection_name, milvus)
    import_to_mongo(ids, answer_dir)


def import_data_csv(collection_name, question_answer_dir):
    """
    导入数据到milvus中
    :param question_answer_dir:
    :param collection_name:
    :return:
    """
    print("开始读取数据")

    question_data, answer_dir = read_data_csv(question_answer_dir, 20000)
    print("开始链接milvus")
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    print("判断是否有同名 table")
    has_table(collection_name, milvus)
    print("导入数据")
    ids = import_to_milvus(question_data, collection_name, milvus)
    print("保存数据")
    import_to_mongo(ids, answer_dir)


def search_in_milvus(collection_name, query_sentence):
    """
    向量搜索数据
    :param collection_name:对应milvus的单元名
    :param query_sentence: 句子
    :return:
    """
    logging.info("start test process ...")
    query_data = [query_sentence]
    try:
        bc = get_bc()

        vectors = bc.encode(query_data)
    except:
        return "bert service disconnect"
    query_list = normaliz_vec(vectors.tolist())
    # connect_milvus_server()
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        # logging.info(status)
    except:
        return "milvus service connection failed"
    try:
        logging.info("start search in milvus...")
        search_params = {'nprobe': 64}
        status, results = milvus.search(collection_name=collection_name, query_records=query_list, top_k=1,
                                        params=search_params)
        if results[0][0].distance < 0.9:
            return "对不起，我暂时无法为您解答该问题"
    except:
        return "milvus service disconnect"


if __name__ == '__main__':
    print("start")
    read_data_csv("medical_questions.csv", 20000)
    import_data_csv("medical_question", "medical_questions.csv")
    print("数据导入结束")