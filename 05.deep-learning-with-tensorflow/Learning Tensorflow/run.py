'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2019-08-25 10:18:46
@LastEditTime: 2019-08-25 10:50:04
'''
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

tf.app.flags.DEFINE_string('mode', 'train', 'train or test')

FLAGS = tf.app.flags.FLAGS

class model():
    def __init__(self):
        self.a = tf.placeholder(tf.float32, [None])
        self.w = tf.Variable(tf.constant(2.0, shape=[1]), name="w")
        b = tf.Variable(tf.constant(0.5, shape=[1]), name="b")
        self.y = self.a * self.w + b

#模型保存为ckpt
def save_model():
    graph1 = tf.Graph()
    with graph1.as_default():
        m = model()
    with tf.Session(graph=graph1) as session:
        session.run(tf.global_variables_initializer())
        update = tf.assign(m.w, [10])
        session.run(update)
        predict_y = session.run(m.y,feed_dict={m.a:[3.0]})
        print(predict_y)

        saver = tf.train.Saver()
        saver.save(session,"model_pb/model.ckpt")


#保存为pb模型
def export_model(session, m):


   #只需要修改这一段，定义输入输出，其他保持默认即可
   # 定义模型输入输出名
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)
    
    # 定义模型保存路径
    export_path = "pb_model/1"
    if os.path.exists(export_path):
        os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))
    
    # 模型保存
    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))

##加载pb模型
def load_pb():
    # 导入pb模型
    session = tf.Session(graph=tf.Graph())
    model_file_path = "pb_model/1"
    meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    
    # 导出模型输出名（保存pb模型时定义的）
    model_graph_signature = list(meta_graph.signature_def.items())[0][1]
    output_tensor_names = [] # tensor_name 是模型定义是给tensor起的别名（未指定时会tensorflow会自动定义）
    # 测试pb模型时只需要用到tensor名
    # 但op_name 与tensor_name 是相关的，也可以通过op_name 得到tensor_name
    #output_op_names = [] # op_name 是在模型保存pb模型时model_signature中名称
    for output_item in model_graph_signature.outputs.items():
        #output_op_name = output_item[0] 
        #output_op_names.append(output_op_name)
        output_tensor_name = output_item[1].name
        output_tensor_names.append(output_tensor_name)
    #print("output_tensor_names: {}; output_op_names: {}".format(output_tensor_names, output_op_names))
    print("load model finish!")
    
    
    sentences = {} # 测试数据
    ## 测试pb模型
    # 模型输入名要与保存pb模型时定义的保持一致
    for test_x in [[1],[2],[3],[4],[5]]:
        sentences["input"] = test_x # key值与保存pb模型时定义保持一致
        feed_dict_map = {}
        for input_item in model_graph_signature.inputs.items():
            #input_op_name = input_item[0]
            input_tensor_name = input_item[1].name
            #print("input_tensor_name: {}; input_op_name: {}".format(input_tensor_name, input_op_name))
            #feed_dict_map[input_tensor_name] = sentences['input_op_name']
            feed_dict_map[input_tensor_name] = test_x
            
        predict_y = session.run(output_tensor_names, feed_dict=feed_dict_map)
        
        print("predict pb y:",predict_y)

def main(argv=None):
    
    if FLAGS.mode == 'train':
        # 1. 训练时保存模型权重
        save_model()
    
        # 2. 定义模型并恢复模型权重
        graph2 = tf.Graph()
        with graph2.as_default():
            m = model()
            saver = tf.train.Saver()
        with tf.Session(graph=graph2) as session:
            saver.restore(session, "model_pb/model.ckpt") #加载ckpt模型
            # 3. 保存为pb模型
            export_model(session, m)
    
    elif FLAGS.mode == 'test':
        # 3. 导入pb模型并测试
        load_pb()
    else:
        print("ERROR")
    
if __name__ == "__main__":
    tf.app.run()
    
