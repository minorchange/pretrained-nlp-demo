import re
import string


def clean_text(text, ):

    def remove_special_characters(text):
        chars_to_rm = string.punctuation
        chars_to_rm = chars_to_rm.replace('-', '')
        chars_to_rm = chars_to_rm.replace('.', '')
        chars_to_rm = chars_to_rm.replace(',', '')
        chars_to_rm += '„“©'
        text = "".join([c for c in text if c not in chars_to_rm])
        return text
    
    def multiple_whitespaces_to_one(text):
        text = ' '.join(text.split())
        return text

    def newline_to_space(text):
        text.replace("\n", " ")
        return text

    text = text.strip(' ') # strip whitespaces at beginning and end of string
    text = remove_special_characters(text)
    text = newline_to_space(text)
    text = multiple_whitespaces_to_one(text)

    return text