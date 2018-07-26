#/usr/bin/python
#-*- coding:utf-8 -*-

import os
import sys
import json
import codecs
import multiprocessing
from threading import Thread

TYPE = sys.argv[2]

# class Readfile(multiprocessing.Process):
#
#     def __init__(self, threadID, file_dir, queue_log, queue_logde):
#         multiprocessing.Process.__init__(self)
#         self.threadID_ = threadID
#         self.file_dir_ = file_dir
#         self.queue_log_ = queue_log
#         self.queue_logde_ = queue_logde
#         self.logid_dict_ = dict()
#         self.logdesignid_dict_ = dict()
#
#     def run(self):
#         print '线程%s 读取文件-----'%self.threadID_
#         for file_name in os.listdir(self.file_dir_):
#             with codecs.open(self.file_dir_ + '/' + file_name, 'r', encoding='utf8') as f:
#                 event_json_list = json.load(f)
#                 for event_json in event_json_list:
#                     try:
#                         logid = str(event_json['log_id'])
#                         logdesignid = str(event_json['log_id']) + '-' + str(event_json['design_id'])
#                         self.logid_dict_[logid] = self.logid_dict_.get(logid, 0) + 1
#                         self.logdesignid_dict_[logdesignid] = self.logdesignid_dict_.get(logdesignid, 0) + 1
#                     except:
#                         pass
#         self.queue_log_.put(self.logid_dict_)
#         self.queue_logde_.put(self.logdesignid_dict_)
#         print '线程%s 读取文件成功 len(logid_dict_)=%d------'%(self.threadID_, len(self.logid_dict_))

class get_id_freq_dict:
    def __init__(self, output_dir, input_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logid_dict = dict()
        self.logdesignid_dict = dict()
        self.threads_ = list()
        self.count = 0

    def check_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def load_data(self):
        for dir in self.input_dir:
            for file_name in os.listdir(dir):
                with codecs.open(dir + '/' + file_name, 'r', encoding='utf8') as f:
                    event_json_list = json.load(f)
                    for event_json in event_json_list:
                        try:
                            logid = event_json.split('#')[0]
                            self.logid_dict[logid] = self.logid_dict.get(logid, 0) + 1
                        except:
                            pass
    
    def print_dict(self):
        print 'execute print_dict---- len(logid_dict) = %d'%len(self.logid_dict)
        if TYPE == 'freq':
            with open(self.output_dir + '/logid_freq', 'w') as f:
                for pair in sorted(zip(self.logid_dict.values(), self.logid_dict.keys()), reverse=True):
                    print>> f, pair[1] + ':' + str(pair[0])

#            with open(self.output_dir + '/logdesignid_freq', 'w') as f:
#                for pair in sorted(zip(self.logdesignid_dict.values(), self.logdesignid_dict.keys())):
#                    print>> f, pair[1] + ':' + str(pair[0])
        elif TYPE == 'all':
            with open(self.output_dir + '/logid_all', 'w') as f:
                for pair in sorted(zip(self.logid_dict.values(), self.logid_dict.keys()), reverse=True):
                    if pair[1] != 'None':
                        print>> f, pair[1]

#            with open(self.output_dir + '/logdesignid_all', 'w') as f:
#                for pair in sorted(zip(self.logdesignid_dict.values(), self.logdesignid_dict.keys()), reverse=True):
#                    if pair[1] != 'None':
#                        print>> f, pair[1]
        elif TYPE == 'top':
            with open(self.output_dir + '/logid_top', 'w') as f:
                for pair in sorted(zip(self.logid_dict.values(), self.logid_dict.keys()), reverse=True):
                    if pair[1] != 'None':
                        print>> f, pair[1]

#            with open(self.output_dir + '/logdesignid_top', 'w') as f:
#                for pair in sorted(zip(self.logdesignid_dict.values(), self.logdesignid_dict.keys()), reverse=True)[:10000]:
#                    if pair[1] != 'None':
#                        print>> f, pair[1]

    def run_process(self):
        self.check_output_dir()
        self.load_data()
        self.print_dict()

if __name__ == '__main__':
    #参数1：输出的路径
    #参数2：数据集类别 比如：freq/all/top
    #参数2-n：输入json的文件路径，可以是多个
    input_dir = ['data/log']
    output_dir = sys.argv[1]
#    for val in sys.argv[3:]:
#        input_dir.append(val)
    get_id_freq_dict_ins = get_id_freq_dict(output_dir, input_dir)
    get_id_freq_dict_ins.run_process()
    print 'GAME OVER'


