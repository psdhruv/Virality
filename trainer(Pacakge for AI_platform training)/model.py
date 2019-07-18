
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf
import shutil
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS=['upv3day', 'msaav10', 'polarity', 'subjectivity', 'num_of_sentences',
       'length', 'num_of_identities', 'other_entities', 'upv3hr', '0', '1',
       '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
       'cat', 'weekday', 'hour']
FEATURES = CSV_COLUMNS[1:]
LABEL = CSV_COLUMNS[0]
#deFAULTS DEFINES datatype of column in tf.decode_csv
DEFAULTS = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],["none"],[0],[0]]


def read_dataset(filename, mode, batch_size = 500):
   def _input_fn():
      def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults = DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL) # remove label from features and store
        return features, label

      # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(filename, shuffle=False)
      # Read lines from text files
      textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset).skip(count=1)
      # Parse text lines as comma-separated values (CSV)
      dataset = textlines_dataset.map(decode_csv)

      # Note:
      # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
      # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = None # loop indefinitely, so limit will be difined by max steps in estimator
          dataset = dataset.shuffle(buffer_size = 70000, seed=2)#100 * batch_size
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs).batch(batch_size)

      return  dataset.make_one_shot_iterator().get_next()
   return _input_fn




def get_train_input_fn(train_dir,training_batch_size):
  return read_dataset(filename=train_dir, mode = tf.estimator.ModeKeys.TRAIN, batch_size=training_batch_size)

def get_valid_input_fn(valid_dir,validation_batch_size):
  return read_dataset(filename=valid_dir, mode = tf.estimator.ModeKeys.EVAL,batch_size=validation_batch_size )







def make_feature_cols():#no y here , must enclose cat columns in indentity or embedding cols for dnn
    categorical_column1=tf.feature_column.categorical_column_with_vocabulary_list(key='cat', vocabulary_list=('blogs','business','dharm','education-and-jobs','entertainment','international','life-style','nari','national','other','regional','sports'))
    categorical_column2=tf.feature_column.categorical_column_with_identity(key='weekday', num_buckets=7)
    categorical_column3=tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column("hour"), boundaries=[6, 12,16, 20])
    input_columns = [tf.feature_column.numeric_column('msaav10'),
                     tf.feature_column.numeric_column('polarity'),
                     tf.feature_column.numeric_column('subjectivity'),
                     tf.feature_column.numeric_column('num_of_sentences'),
                     tf.feature_column.numeric_column('length'),
                     tf.feature_column.numeric_column('num_of_identities'),
                     tf.feature_column.numeric_column('other_entities'),
                     tf.feature_column.numeric_column('upv3hr'),
                     tf.feature_column.numeric_column('0'),
                     tf.feature_column.numeric_column('1'),
                     tf.feature_column.numeric_column('2'),
                     tf.feature_column.numeric_column('3'),
                     tf.feature_column.numeric_column('4'),
                     tf.feature_column.numeric_column('5'),
                     tf.feature_column.numeric_column('6'),
                     tf.feature_column.numeric_column('7'),
                     tf.feature_column.numeric_column('8'),
                     tf.feature_column.numeric_column('9'),
                     tf.feature_column.numeric_column('10'),
                     tf.feature_column.numeric_column('11'),
                     tf.feature_column.numeric_column('12'),
                     tf.feature_column.numeric_column('13'),
                     tf.feature_column.numeric_column('14'),
                     tf.feature_column.embedding_column(categorical_column1,dimension=10),
                     
                     tf.feature_column.indicator_column(categorical_column2),
                     tf.feature_column.indicator_column(categorical_column3)]
    return input_columns

def serving_input_fn():#actual datatype in data for serving
    feature_placeholders =  {
        column.name: tf.placeholder(tf.float32, [None]) for column in list(make_feature_cols())[:23]
    }

    feature_placeholders["cat"]=tf.placeholder(tf.string, [None])
    feature_placeholders["weekday"]=tf.placeholder(tf.int32, [None])
    feature_placeholders["hour"]=tf.placeholder(tf.int32, [None])
    
    
    features = feature_placeholders#preprocessing step
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def my_metric(labels, predictions):
    pred_values = predictions['predictions']
    return {'metric1': tf.metrics.root_mean_squared_error(labels, pred_values)}





def train_and_evaluate(args):
    run_config = tf.estimator.RunConfig( save_checkpoints_secs = 5, keep_checkpoint_max = 4, tf_random_seed=100)
    num_steps = 250  #(70000 / args['train_bs']) / (args['learning_rate']*2)  # if learning_rate=0.01, 50 epochs, use this when hptuning

    tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    estimator = tf.estimator.DNNRegressor(
        config = run_config,
        model_dir = args['output_dir'],
        feature_columns = make_feature_cols(),
        hidden_units =[100,90,81,73],#[max(2, int(args['first_layer_size'] * args['scale_factor']**i)) for i in range(args['num_layers'])]use this when hptuning
        optimizer=tf.train.AdagradOptimizer(args['learning_rate']))
    estimator = tf.contrib.estimator.add_metrics(estimator, my_metric)
    train_spec = tf.estimator.TrainSpec(
        input_fn =get_train_input_fn(args["train_dir"], args["train_bs"]),
         max_steps = num_steps)
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = get_valid_input_fn(args["valid_dir"], args["eval_bs"]),
        steps = None, #none will evaluate whole eval data in each test
        start_delay_secs = args["eval_delay_secs"],
        throttle_secs = args["throttle_secs"],
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
