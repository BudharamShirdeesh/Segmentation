import requests
import re
import json
import csv
from bs4 import BeautifulSoup
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate



# Define the CSV file path
csv_file = 'finaloutput1.csv'

def remove_svara(text):
    """ Removes svaras and specified characters """
    new_text = []
    for char in text:
        if '\u0951' <= char <= '\u0954':  # Check for Devanagari vowel signs (U+0951 to U+0954)
            continue
        # To remove zero width joiner character and other specified characters
        elif char in ['\u200d', '\u200c', '\u1CD6', '\u1CD5', '\u1CE1', '\uA8EB', '\u030E', '\u035B', '\u0324', '\uA8E2', '\u1CB5', '\uA8EC', '\u0331', '\u1CB6', '\uA8F1', '\u032B', '\uA8EF', '\uA8E3', '\uA8E1', '\u0308', '\u030D', '\u200D']:
            continue
        new_text.append(char)
    
    modified_text = "".join(new_text)
    return modified_text

def process_sentence(sentence):
    words = sentence.split()
    processed_words = [remove_svara(word) for word in words]
    processed_sentence = " ".join(processed_words)
    return processed_sentence

def fetch_segmentation_details(sentence):
    url = "https://sanskrit.uohyd.ac.in/cgi-bin/scl/MT/prog/sandhi_splitter/sandhi_splitter.cgi"
    params = {
        "word": sentence,
        "outencoding": "I",
        "encoding": "IAST",
        "mode": "sent"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful
        response.encoding = 'utf-8-sig'
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser', from_encoding='utf-16')
        
        # Find the div with id 'finalout'
        finalout_div = soup.find('div', {'id': 'finalout'})
        
        # Extract the text content from the div
        if finalout_div:
            div_text = finalout_div.get_text(separator='\n').strip()
            parts = div_text.split('\n')
            segmentation = parts[0].strip().replace('[', '').replace(']', '')
        else:
            segmentation = "No segmentation found."
        
        # Split the segmentation by '-' and ' ' to get individual words
        segmented_words = split_segmentation(segmentation)
        return segmentation, segmented_words
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch segmentation from {url}. {e}")
        return None, []
    except Exception as e:
        print(f"Error: An unexpected error occurred during segmentation: {e}")
        return None, []

def split_segmentation(segmentation):
    segmented_words = segmentation.split('-')
    segmented_words = [word.strip() for segment in segmented_words for word in segment.split()]
    segmented_words_iast = [transliterate(word, sanscript.DEVANAGARI, sanscript.IAST) for word in segmented_words]
    return segmented_words_iast

def clean_meaning_text(text):
    cleaned_text = re.sub(r'\bmfn\.|\bm\.\b|\bf\.\b|\bn\.\b|\bind\.\b|\bE\.\b|\bp\.\b', '', text)
    cleaned_text = re.sub(r'\(\-.*?\)', '', cleaned_text)  # Remove (-laḥ-lā-laṃ)
    cleaned_text = re.sub(r'^\d+\.\s*', '', cleaned_text)  # Remove leading numbers
    return cleaned_text.strip()

def get_list_meaning_word(seg_words):
    meanings_list = []

    for word in seg_words:
        word_meanings = []
        url = f"https://ambuda.org/tools/dictionaries/mw,shabdasagara,apte/{word}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'utf-8-sig'

            soup = BeautifulSoup(response.text, 'html.parser')
            divs = soup.find_all('div', class_='my-4', attrs={'x-show': 'show'})

            try:
                div_items_0 = divs[0].find('ul').find_all('li', class_='dict-entry mw-entry')
                dive_text_0 = [clean_meaning_text(li_tag.get_text(strip=True)) for li_tag in div_items_0 if li_tag.get_text(strip=True)]
                text_0_trans = [transliterate(text, sanscript.DEVANAGARI, sanscript.IAST) for text in dive_text_0 if text.strip()]
                word_meanings.extend(text_0_trans)
            except (IndexError, AttributeError):
                print(f"Error: Unable to find Monier-Williams Sanskrit-English Dictionary (1899) data for '{word}'.")

            try:
                div_items_1 = divs[1].find_all('div')
                dive_text_1 = [clean_meaning_text(item.get_text(strip=True)) for item in div_items_1 if item.get_text(strip=True)]
                text_1_trans = [transliterate(text, sanscript.DEVANAGARI, sanscript.IAST) for text in dive_text_1 if text.strip()]
                word_meanings.extend(text_1_trans)
            except (IndexError, AttributeError):
                print(f"Error: Unable to find Shabda-Sagara (1900) data for '{word}'.")

            try:
                apte_meanings = []
                for tag in divs[2].find_all('b'):
                    if tag.text.strip() != '—':
                        text1 = clean_meaning_text(tag.text.strip())
                        sibling = tag.find_next_sibling()
                        text2 = clean_meaning_text(tag.next_sibling.strip()) + ' '
                        while sibling and sibling.name != 'div':
                            if sibling.name is None:
                                text2 += " "
                            elif sibling.name == 'span':
                                IAST_text = transliterate(sibling.text.strip(), sanscript.DEVANAGARI, sanscript.IAST)
                                text2 += IAST_text + ' ' + clean_meaning_text(sibling.next_sibling.strip())
                            else:
                                text2 += clean_meaning_text(sibling.text.strip()) + ' ' + clean_meaning_text(sibling.next_sibling.strip())
                            sibling = sibling.find_next_sibling()
                        apte_meanings.append(text2)
                word_meanings.extend(apte_meanings[:-1])
            except (IndexError, AttributeError):
                print(f"Error: Unable to find Apte-Practical Sanskrit-English Dictionary (1890) data for '{word}'.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to fetch data from {url}. {e}")

        meanings_list.append({
            'pada': word,
            'meanings': word_meanings  # Return meanings as a comma-separated string
        })
    
    return meanings_list

def load_csv_data():
    data = []
    with open(csv_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def get_details(sentence):
    # Process sentence to remove swaras
    processed_sentence = process_sentence(sentence)
    
    # Fetch segmentation details
    segmentation, segmented_words = fetch_segmentation_details(processed_sentence)
    
    if segmentation:
        # Get meanings for each segmented word
        meanings_list = get_list_meaning_word(segmented_words)
        
        # Load CSV data
        csv_data = load_csv_data()
        
        # Prepare a list of meanings in the required format
        meanings_output = []
        for meaning in meanings_list:
            csv_meaning = None
            for row in csv_data:
                if meaning['pada'] == row['form']:
                    csv_meaning = row['meaning']
                    break
            # Construct the output format
            meanings_text = {
                f"the list of possible meanings of {meaning['pada']} are": [meaning['meanings']]
            }
            meanings_output.append(meanings_text)
        
        # Prepare segmented_words as a comma-separated list
        segmented_words_csv = ",".join(segmented_words)
        
        return {
            'sentence': processed_sentence,
            'segmented_words': segmented_words_csv,
            'meanings': meanings_output
        }
    else:
        return {
            "error": "Segmentation failed. Please check your input."
        }

if __name__ == "__main__":
    # Example usage
    sentence = "pudgaladharmanairātmyapratipādanaṃ punaḥ kleśajñeyāvaraṇaprahāṇārtham"
    details = get_details(sentence)
    
    # Print the final output as JSON
    if 'error' in details:
        print(json.dumps(details, indent=4, ensure_ascii=False))
    else:
        print(json.dumps(details, indent=4, ensure_ascii=False))
