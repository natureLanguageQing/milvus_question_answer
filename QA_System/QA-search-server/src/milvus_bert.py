import logging
from functools import reduce

import numpy as np
import src.config as config
import src.pg_operating as pg_operating
from bert_serving.client import BertClient
from milvus import *
from pymongo import MongoClient
import pandas as pd
bc = BertClient(ip=config.BERT_CLIENT_HOST)

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


conn = MongoClient(config.MONGO_HOST, config.MONGO_PORT)
db = conn.medicalQa

my_set = db.open

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def import_to_mongo(ids, answer_file):
    """
    数据导入mongo数据库中
    :param ids:
    :param answer_file:
    :return:
    """
    my_set.save({"ids": ids, "answer_file": answer_file})


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
    vectors = bc.encode(data)
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


def read_data_csv(file_dir,top_k):
    """
    读取txt数据
    :param file_dir:
    :return:
    """
    data = pd.read_csv(file_dir).drop_duplicates().dropna().values.tolist()[:top_k]
    for i in data:
        print(i)
    return data


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


def import_data_csv(collection_name, qustion_answer_dir):
    """
    导入数据到milvus中
    :param collection_name:
    :param question_dir:
    :param answer_dir:
    :return:
    """
    question_data = read_data_csv(qustion_answer_dir,20000)
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    has_table(collection_name, milvus)
    ids = import_to_milvus(question_data, collection_name, milvus)
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

    try:
        conn = pg_operating.connect_postgres_server(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)
        cur = conn.cursor()
    except:
        return "postgres service connection failed"
    try:
        logging.info("start search in pg ...")
        rows = pg_operating.search_in_pg(conn, cur, results[0][0].id, collection_name)
        out_put = rows[0][1]
        return out_put
    except:
        return "postgres service disconnect"
    finally:
        conn.close()


if __name__ == '__main__':
    create_milvus_table("medical_qa", "../../../data/medical_questions.csv")
