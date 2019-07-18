import argparse
import json
import os
import shutil

# for python3
from . import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_dir',
        help = 'GCS or local path to training data',
        default= "gs://data-etl-process1/final_data2/df_train.csv"
    )
    parser.add_argument(
        '--valid_dir',
        help = 'GCS or local path to validation data',
        default= "gs://data-etl-process1/final_data2/df_valid.csv"
    )
    parser.add_argument(
        '--train_bs',
        help = 'Batch size for training steps',
        type = int,
        default = 70000
    )
    parser.add_argument(
        '--eval_bs',
        help = 'Batch size for evaluation steps',
        type = int,
        default = 5000
    )
#     parser.add_argument(
#         '--num_train_steps',
#         help = 'Steps to run the training job for',
#         type = int
#         required = True
#     )
#     parser.add_argument(
#         '--eval_steps',
#         help = 'Number of steps to run evalution for at each checkpoint',
#         default = 10,
#         type = int
#     )
#     parser.add_argument(
#         '--eval_data_paths',
#         help = 'GCS or local path to evaluation data',
#         required = True
#     )
    # Training arguments
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        default = 'gs://data-etl-process1/final_model2/'
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'gs://data-etl-process1/final_model2/'
    )

    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--throttle_secs',
        help = 'Seconds between evaluations',
        default = 10,
        type = int
    )
    parser.add_argument(
      '--learning_rate',
      type = float, 
      default = 0.005
    )
#     parser.add_argument(
#         '--first-layer-size',
#         help='Number of nodes in the first layer of the DNN',
#         default=10,
#         type=int)
#     parser.add_argument(
#         '--num-layers',
#         help='Number of layers in the DNN',
#         default=2,
#         type=int)
#     parser.add_argument(
#         '--scale-factor',
#         help='How quickly should the size of the layers in the DNN decay',
#         default=0.5,
#         type=float)
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
#     arguments['output_dir'] = os.path.join(
#       arguments['output_dir'],
#       json.loads(
#           os.environ.get('TF_CONFIG', '{}')
#       ).get('task', {}).get('trial', '')
#   )

    # Run the training job
    # Run the training
    #shutil.rmtree(arguments['output_dir'], ignore_errors=True) # start fresh each time
    model.train_and_evaluate(arguments)





    
