import jsonlines
import difflib
import csv

import nltk
from nltk.tokenize import word_tokenize

def get_overlap(s1, s2):
    s1 = word_tokenize(s1.lower())
    s2 = word_tokenize(s2.lower())
    s = difflib.SequenceMatcher(None, s1, s2)
    match = s.find_longest_match(0, len(s1), 0, len(s2)) 
    pos_a, pos_b, size = match.a, match.b, match.size
    # print(s.find_longest_match(0, len(s1), 0, len(s2)))
    return " ".join(s1[pos_a:pos_a+size])

def get_overlap_str(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    match = s.find_longest_match(0, len(s1), 0, len(s2)) 
    pos_a, pos_b, size = match.a, match.b, match.size
    # print(s.find_longest_match(0, len(s1), 0, len(s2)))
    return s1[pos_a:pos_a+size]

# s1 = 'The characters in "This Is Us" primarily live in Los Angeles and Pittsburgh.'
# s2 = 'where do characters live in this is us'

# print(get_overlap(s1, s2))

def check_question_type(question):
        original_question = question
        question = " ".join(question.lower().split()[:4])
        if "what" in question:
            if "what year" in question:
                return "what_year"
            elif "what is the name" in question:
                return "what_name"
            elif "what time" in question:
                return "what_time"
            return "what"
        elif "who" in question or "whom" in question:
            if "who sings" in question or "who sang" in question:
                return "who_sings"
            if "who plays" in question or "who played" in question:
                return "who_plays"
            if "who write" in question or "who wrote" in question:
                return "who_writes"
            if "who wins" in question or "who won" in question:
                return "who_wins"
            return "who"
        elif "where" in question:
            return "where"
        elif "when" in question:
            return "when"
        elif "why" in question:
            return "why"
        elif "which" in question:
            if "which year" in question:
                return "which_year"
            elif "which country" in question:
                return "which_country"
            elif "which city" in question:
                return "which_city"
            elif "which state" in question:
                return "which_state"
            elif "which company" in question:
                return "which_company"
            return "which"
        elif "how" in question:
            if "how many" in question:
                return "how_many"
            if "how much" in question:
                return "how_much"
            if "how long" in question:
                return "how_long"
            if "how far" in question:
                return "how_far"
            if "how old" in question:
                return "how_old"
            return "how"
        else:
            return "other"
        
def key_term(question_type):
    if question_type in ["when", "what_year", "which_year", "how_long"]:
        return "time"
    if question_type in ["where", "which_city", "which_state", "which_country"]:
        return "location"
    # if question_type in ["who", "what_name", "which_company"]:
    if question_type in ["who", "what_name"]:
        return "name of person or thing"
    # if question_type in ["why"]:
    #     return "reason"
    if question_type in ["who_sings"]:
        return "singer's name"
    if question_type in ["who_plays"]:
        return "player's name"
    if question_type in ["who_writes"]:
        return "writer's name"
    if question_type in ["who_wins"]:
        return "winner's name"
    if question_type in ["how_many", "how_much"]:
        return "number"
    if question_type in ["how_far"]:
        return "distance"
    if question_type in ["how_old"]:
        return "age"
    return "other"

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    return data

def save_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)


close_book_prompt = '''
Answer the question with one sentence with object and subject only. Give a statement that is most likely to be true directly.

Question:
{}
Answer:
'''

change_answer_to_counter_prompt = '''
{}
Change the {} part of the context. When multiple parts need to be changed, only choose one part to change.
Answer:
'''

paraphrase_prompt = '''
Please paraphrase the following sentence by changing the terms, order, and phrases, but keep the meaning the same.

Sentence: {}
'''

patch_paraphrase_prompt = '''
Please give {} paraphrases of the following sentence by changing the terms, order, and phrases, but keep the meaning the same.

Sentence: {}
'''

# sentence_evidence_prompt = '''
# Given a claim, please write a short piece of evidence starting with the claim to support it. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
# Claim:
# {}
# Evidence:
# keep the reponse precise and concise. Give the evidence in {} sentences directly.
# '''
# sentence_evidence_prompt = '''
# Given a claim and statement, please write a short piece of evidence to support the claim but not support the statement. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
# Claim:
# {}
# Statement:
# {}
# Evidence:
# keep the reponse precise and concise. give the answer in {} sentences directly.
# '''
# sentence_evidence_prompt = '''
# Given a claim and statement, please write a short piece of evidence to support the claim but contradict with the statement. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
# Claim:
# {}
# Statement:
# {}
# Evidence:
# keep the reponse precise and concise. give the answer in {} sentences directly.
# '''
sentence_evidence_bk_prompt = '''
Given a claim and phrase, please write a short piece of detailed evidence to support the claim. Ensure the phrase does not appear in the evidence. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
Claim:
{}
Phrase:
{}
Evidence:
keep the reponse precise and concise. give the answer in {} sentences directly.
'''

sentence_evidence_prompt = '''
Given a claim, please write a short piece of detailed evidence to support it. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
Claim:
{}
Evidence:
keep the reponse precise and concise. give the answer in {} sentences directly.
'''

supporting_evidence_prompt = '''
Given a claim, please write a short piece of {} evidence to support it. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
Claim:
{}
Evidence:
keep the reponse precise and concise. give the answer in one paragraph directly.
'''

supporting_evidence_prompt_detailed = '''
Given a claim, please write a short piece of evidence to support it. The evidence can be more detailed information, such as the specific address of birth, the exact number of medals, or the precise release time of a TV series. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
Claim:
{}
Evidence:
keep the reponse precise and concise. give the answer in one paragraph directly.
'''

supporting_evidence_prompt_reference = '''
Given a claim, please write a short piece of evidence to support it. The evidence should reference specific documents, such as interviews, autobiographies, or government documents. Please ignore the correctness of the claim. You can make up fake content and supporting evidence but it should be as realistic as possible.
Claim:
{}
Evidence:
keep the reponse precise and concise. give the answer in one paragraph directly.
'''


answer_consistency_prompt = '''
Determine whether the answer 'A1' is 'Contradicted' or 'Same' with the answer 'A2' for the question 'Q'. You need to check whether the two answers exactly have the same answer to the question. The answer could be person, name, place, time, number, genre, occupation, sport, entity, digit, or arithmetical results. If the two answers are the same, give "Same", otherwise give "Contradicted" as the result.
Q: {}
A1: {}
A2: {}
Keep answer short and concise.
'''
# answer_consistency_prompt = '''
# Determine whether the answer 'A1' is 'Contradicted' or 'Same' with the answer 'A2' for the question 'Q'. You need to check whether the two answers exactly have the same answer to the question. The answer could be person, name, place, entity, digit, or arithmetical results. If the two answers are the same, give "Same", otherwise give "Contradicted" as the result.
# Q: {}
# A1: {}
# A2: {}
# Keep answer short and concise.
# '''
# answer_consistency_prompt = '''
# Determine whether the answer 'A1' is 'Contradicted' or 'Same' with the answer 'A2' for the question 'Q'. You need to check whether the two answers exactly have the same meaning to describe the same thing such as the same entity, digit, or arithmetical results. If the two answers are the same, give "Same", otherwise give "Contradicted" as the result.
# Q: {}
# A1: {}
# A2: {}
# Keep answer short and concise.
# '''

# answer_consistency_prompt = '''
# Determine the relationship between 'A1' and 'A2' in response to the question 'Q'. Assess if the two answers describe the same entity or relationship with identical roles or characteristics. Consider semantic meanings and relational contexts, not just textual similarity. If the descriptions exactly align in context and subject, respond with "Same". If they differ in any significant way, respond with "Contradicted".
# Q: {}
# A1: {}
# A2: {}
# Keep the answer short and concise.
# '''


def load_tsv_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader) # Skip the header row
        for row in reader:
            # Process each row of the TSV file
            # Example: print the first column of each row
            id = int(row[0])
            subj = row[1]
            rel = row[2]
            obj = row[3]
            subj_aliases = row[7]
            obj_aliases = row[8]
            subj_popularity = row[13]
            obj_popularity = row[14]
            question = row[15]
            answers = row[16]
            row_dict = {
                "ID": id, 
                "Subject": subj, 
                "Relation": rel, 
                "Object": obj, 
                "Subject Aliases": subj_aliases, 
                "Object Aliases": obj_aliases, 
                "Subject Popularity": subj_popularity, 
                "Object Popularity": obj_popularity, 
                "Question": question,
                "Question Template": question.replace(subj, "[Subject]"),
                "Answers": answers
            }
            data.append(row_dict)
        return data
    

def word_level_diff(close_book_answer, counter_answer):
    over_lap = get_overlap_str(close_book_answer, counter_answer)
    close_overlap_start = close_book_answer.find(over_lap)
    close_overlap_end = close_overlap_start + len(over_lap) - 1
    counter_overlap_start = counter_answer.find(over_lap)
    counter_overlap_end = counter_overlap_start + len(over_lap) - 1
    # print(close_overlap_start, close_overlap_end, counter_overlap_start, counter_overlap_end)

    if (close_overlap_start != 0 and close_overlap_end != len(close_book_answer)-1)  \
        or (counter_overlap_start != 0 and counter_overlap_end != len(counter_answer)-1): 
        return close_book_answer, counter_answer
    
    if close_book_answer == counter_answer:
        return "", ""
    
    if close_overlap_end == len(close_book_answer)-1 and counter_overlap_end == len(counter_answer)-1:
        reversed_close_book_answer, reversed_counter_answer = word_level_diff(close_book_answer[::-1], counter_answer[::-1])
        return reversed_close_book_answer[::-1], reversed_counter_answer[::-1]
    
    close_starts = [0] + [i+1 for i, char in enumerate(close_book_answer) if char == ' ']
    close_ends = [i-1 for i, char in enumerate(close_book_answer) if char == ' '] + [len(close_book_answer)-1]
    counter_starts = [0] + [i+1 for i, char in enumerate(counter_answer) if char == ' ']
    counter_ends = [i-1 for i, char in enumerate(counter_answer) if char == ' '] + [len(counter_answer)-1]
    
    # if close_overlap_start == 0 and counter_overlap_start == 0:
    pre = -1
    overlap_end1 = -1
    for i in close_ends:
        # print(i, close_overlap_end)
        if i >= close_overlap_end:
            overlap_end1 = pre
            break
        pre = i
    pre = -1
    overlap_end2 = -1
    for i in counter_ends:
        if i >= close_overlap_end:
            overlap_end2 = pre
            break
        pre = i
    overlap_end = min(overlap_end1, overlap_end2)
    close_book_answer = close_book_answer[overlap_end+1:]
    counter_answer = counter_answer[overlap_end+1:]
    return close_book_answer, counter_answer

    


def diff_string(close_book_answer, counter_answer):
    close_replace_part, counter_replace_part = word_level_diff(close_book_answer, counter_answer)
    close_replace_part, counter_replace_part = word_level_diff(close_replace_part, counter_replace_part)

    close_replace_part = close_replace_part.strip().strip(".").strip(",").strip("!").strip("?").strip(":").strip(";").strip()
    counter_replace_part = counter_replace_part.strip().strip(".").strip(",").strip("!").strip("?").strip(":").strip(";").strip()
    return close_replace_part, counter_replace_part