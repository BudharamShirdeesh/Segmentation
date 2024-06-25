import sys
import os
import json
import csv
import re
import tensorflow as tf
from bs4 import BeautifulSoup
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Assuming main_script.py is located in segmentation folder and needs to access data.code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'code')))

# Import from local modules
from segm.seg_meaning import (
    process_sentence,
    fetch_segmentation_details,
    clean_meaning_text,
    get_list_meaning_word,
    split_segmentation
)
from helper_functions import analyze_text
import data_loader
from configuration import config  # Import config from configuration

# Define the CSV file path
csv_file = 'finaloutput1.csv'

# Define lists of special characters
svaras = ['\uA8E1', '\uA8E2', '\uA8E3', '\uA8E4', '\uA8E5', '\uA8E6', '\uA8E7', '\uA8E8', '\uA8E9', '\uA8E0', '\uA8EA', '\uA8EB', '\uA8EC', '\uA8EE', '\uA8EF', '\u030D', '\u0951', '\u0952', '\u0953', '\u0954', '\u0945']
special_characters = ['\uf15c', '\uf193', '\uf130', '\uf1a3', '\uf1a2', '\uf195', '\uf185', '\u200d', '\u200c', '\u1CD6', '\u1CD5', '\u1CE1', '\u030E', '\u035B', '\u0324', '\u1CB5', '\u0331', '\u1CB6', '\u032B', '\u0308', '\u030D', '\u0942', '\uF512', '\uF693', '\uF576', '\uF11E', '\u1CD1', '\u093C', '\uF697', '\uF6AA', '\uF692']
chandrabindu = ['\u0310']

def remove_svaras(text):
    """ Removes svaras and special characters from the text """
    return ''.join([char for char in text if char not in (svaras + special_characters)])

def handle_input(input_text, input_encoding):
    """ Modifies input based on the requirement of the Heritage Engine """
    modified_input = remove_svaras(input_text)

    # Replace special characters with "." since Heritage Segmenter
    # does not accept special characters except "|", "!", "."
    modified_input = re.sub(r'[$@#%&*()\[\]=+.:;"}{?/,\\।॥]', ' ', modified_input)
    
    if input_encoding != "RN":
        modified_input = modified_input.replace("'", " ")

    # Handle chandrabindu based on input_encoding
    if input_encoding == "DN":
        modified_input = modified_input.replace(chandrabindu[0], "म्" if modified_input[-1] == chandrabindu[0] else "ं")
    elif input_encoding == "IAST":
        modified_input = re.sub(r'M$', 'm', modified_input)
        modified_input = re.sub(r'\.m$', '.m', modified_input)

    return modified_input

def analyze_with_external_tool(word):
    """ Calls an external tool to analyze the input word """
    # Path to the input and output files
    input_file_path = r'C:/Users/shird/OneDrive/Desktop/intern/finalseg/data/code/input_iast.txt'
    output_file_path = input_file_path + '.unsandhied'

    # Write the word to the input file
    with open(input_file_path, 'w', encoding='utf-8') as file:
        file.write(word)

    # Load data model partially
    with data_loader.DataLoader(r'C:/Users/shird/OneDrive/Desktop/intern/finalseg/data/input', config, load_data_into_ram=False, load_data=False) as data:
        graph_pred = tf.compat.v1.Graph()
        with graph_pred.as_default():
            sess = tf.compat.v1.Session(graph=graph_pred)
            with sess.as_default():
                # Restore saved values
                print('\nRestoring...')
                model_dir = os.path.normpath(os.path.join(os.getcwd(), config['model_directory']))
                print(f"Model directory: {model_dir}")  # Print model directory for debugging
                tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], model_dir)
                print('Ok')

                # Tensor names based on trained model architecture
                x_ph = graph_pred.get_tensor_by_name('inputs:0')
                split_cnts_ph = graph_pred.get_tensor_by_name('split_cnts:0')
                dropout_ph = graph_pred.get_tensor_by_name('dropout_keep_prob:0')
                seqlen_ph = graph_pred.get_tensor_by_name('seqlens:0')
                predictions_ph = graph_pred.get_tensor_by_name('predictions:0')

                # Analyze the input text
                output = analyze_text(input_file_path, output_file_path, predictions_ph, x_ph, split_cnts_ph, seqlen_ph, dropout_ph, data, sess, verbose=True)
                
                # Print output to terminal if not None
                if output is not None:
                    print("\nOutput:")
                    return output
                else:
                    print("No output returned from analyze_text()")

def load_csv_data():
    """ Loads data from CSV file """
    data = []
    with open(csv_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def ss(sentence):
    """ Main function for processing the input sentence """
    # Step 1: Process the sentence to remove swaras
    processed_sentence = handle_input(sentence, "IAST")
    print("Input Sentence:", sentence)  # Debug: Print processed sentence
    
    # Step 2: Fetch segmentation details from original function
    segmentation1, segmented_words1 = fetch_segmentation_details(processed_sentence)
    print("Segmented Words 1 (from fetch_segmentation_details):", segmented_words1)  # Debug: Print segmented words 1
    
    # Step 3: Fetch and process segmentation details from tool
    analyzed_words = {}
    segmented_words2 = []
    for word in segmented_words1:
        if '?' in word:
            analyzed_word = analyze_with_external_tool(word)
            if analyzed_word:
                analyzed_word = analyzed_word.replace('ḷ', '')  # Remove 'ḷ' from analyzed words
                analyzed_words[word] = analyzed_word  # Store in dictionary with original word as key
                segmented_words2.append(analyzed_word)  # Use analyzed result as segmentation 2 without 'ḷ'
        else:
            segmented_words2.append(word)
    
    # Step 4: Generate final segmented words in the order of input sentence
    final_segmented_words = []
    for word in segmented_words1:
        if word in analyzed_words:
            final_segmented_words.append(analyzed_words[word])
        else:
            final_segmented_words.append(word)
    
    # Convert segmented_words1, segmented_words2, and final_segmented_words to comma-separated lists
    segmented_words1_str = ", ".join(segmented_words1)
    segmented_words2_str = ", ".join(segmented_words2)
    final_segmented_words_str = ", ".join(final_segmented_words)
    
    # Split final_segmented_words_str by ", " and handle special case for "-"
    final_segmented_words_split = []
    for word in final_segmented_words_str.split(", "):
        if "-" in word:
            final_segmented_words_split.extend(word.split("-"))
        else:
            final_segmented_words_split.append(word)
    
    # Step 5: Get detailed meanings for final segmented words from dictionary.csv and online dictionaries
    meanings_list = get_list_meaning_word(final_segmented_words_split)
    csv_data = load_csv_data()
    meanings_output = []
    
    for meaning in meanings_list:
        csv_meaning = next((row['meaning'] for row in csv_data if row['form'] == meaning['pada']), None)
        
        # Construct the output format
        meanings_text = {
            f"the list of possible meanings of {meaning['pada']} are": meaning['meanings']
        }
        
        if csv_meaning:
            meanings_text[f"the list of possible meanings of {meaning['pada']} are"].append(csv_meaning)
        
        meanings_output.append(meanings_text)
    
    return segmented_words1_str, segmented_words2_str, list(analyzed_words.values()), final_segmented_words_split, meanings_output

# Example usage
if __name__ == "__main__":
    input_sentence = "dhrtarastra uvaca dharmaksetre kuruksetre samaveta yuyutsavah mamakah pandavas caiva kim akurvata Samjaya  "
    segmented_words1, segmented_words2, analyzed_words, final_segmented_words, meanings_output = ss(input_sentence)
    
    # Print the results in the desired format
    output = {
        "input sentence": input_sentence,
        "segmented words 1": segmented_words1,
        "segmented words 2": segmented_words2,
        "analyzed words": analyzed_words,
        "final segmented words": ", ".join(final_segmented_words),
        "meanings": meanings_output
    }
    
    # Print the output as JSON
    print(json.dumps(output, indent=4, ensure_ascii=False))
