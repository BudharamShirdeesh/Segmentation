import tensorflow as tf
import configuration
from helper_functions import analyze_text
import data_loader
import sys
import os

def main():
    # if len(sys.argv) < 2:
    #     print('At least one argument is required!')
    #     print(' 1st argument: path of the input text')
    #     print(' 2nd argument (optional): path for the result. If empty = [input path].unsandhied')
    #     exit()

    # path_in = sys.argv[1]
    path_in = r'C:/Users/shird/OneDrive/Desktop/intern/segmentation/data/code/input_iast.txt'
    if not os.path.exists(path_in):
        print(f'Input file {path_in} not found')
        exit()
    path_out = path_in + '.unsandhied' if len(sys.argv) < 3 else sys.argv[2]

    config = configuration.config

    # Load a partial data model
    with data_loader.DataLoader(r'C:/Users/shird/OneDrive/Desktop/intern/finalseg/data/input', config, load_data_into_ram=False, load_data=False) as data:
        graph_pred = tf.compat.v1.Graph()
        with graph_pred.as_default():
            sess = tf.compat.v1.Session(graph=graph_pred)
            with sess.as_default():
                # Restore saved values
                print('\nRestoring...')
                model_dir = os.path.normpath(os.path.join(os.getcwd(), config['model_directory']))
                tf.compat.v1.saved_model.load(
                    sess,
                    [tf.saved_model.SERVING],
                    model_dir
                )
                print('Ok')

                # Tensor names based on the trained model architecture
                x_ph = graph_pred.get_tensor_by_name('inputs:0')
                split_cnts_ph = graph_pred.get_tensor_by_name('split_cnts:0')
                dropout_ph = graph_pred.get_tensor_by_name('dropout_keep_prob:0')
                seqlen_ph = graph_pred.get_tensor_by_name('seqlens:0')
                predictions_ph = graph_pred.get_tensor_by_name('predictions:0')

                # Analyze the input text
                output = analyze_text(path_in, path_out, predictions_ph, x_ph, split_cnts_ph, seqlen_ph, dropout_ph, data, sess, verbose=True)
                
                # Print the output directly to terminal if it's not None
                if output is not None:
                    print("\nOutput:")
                    print(output)
                    return output
                else:
                    print("No output returned from analyze_text()")

if __name__ == "__main__":
    main()
