import collections
import pandas as pd
import course_scraper
from nltk.corpus import stopwords
import string
import re

DESCRIPTION = "Description"
COURSE_PREFIX = "Course Prefix"

words_to_remove = ["lectures", "per", "two", "and/or", "``", "''", "laboratory", "course", "courses", "work",
                   "students", "units", "total", "selected", "may", "major", "'s", "quarter", "and/or", "report", "undergraduate", "format",
                   "laboratory", "limited", "topics", "fulfills", "including", "topic", "catalogs", "list", "earlier", "overview", "impact",
                   "required", "open", "study", "class", "grading", "credit/no", "individual", "kine", "new", "within", "offered",
                   "laboratories", "include", "use", "using", "used", "basic", "student", "current", "related", "practice",
                   "online", "examination", "formal", "quality", "one", "time", "must", "maximum", "hours", "effects"]
ge_areas = ["a", "b", "c", "d", "e", "f",
            "area", "areas", "uscp", "upper-division"]
year = ["2017-19", "2019-20"]

stopwords_to_remove = ["ge", "credit", "class", "topics", "course", "following", "student", "units", "section", "study", "k", "unit", "week", "used",
                       "division", "catalogs", "graduate", "selected", "courses", "may", "majors", "format", "emphasis", "area", "hours", "emphasized",
                       "non", "based", "application", "applications", "classroom", "introduction", "students", "crosslisted", "focus", "methods", "completion",
                       "required", "implementation", "u", "better", "part", "fields", "completed", "taken", "well", "grade", "present", "basic", "etc"
                       "graduates", "variety", "context", "presented", "instruction", "quarter", "projects", "meet", "fulfills", "enroll", "enrollment",
                       "requirement", "studies", "surveys", "planning", "discussion", "assessment", "role", "field", "preparation", "principles", "evaluation",
                       "techniques", "selection", "practices", "concepts", "faculty", "theories", "issues", "paid", "usually", "quarters", "independent",
                       "fundamentals", "project", "senior"]


def generate_ge_prefixes():
    letters = ["a", "b", "c", "d", "e", "f"]
    numbers = list(range(1, 8))

    pairs = []
    for letter in letters:
        for num in numbers:
            pairs.append(letter + str(num))
    return pairs


ge_prefixes = generate_ge_prefixes()
filter_set = set(stopwords.words('english'))
filter_set.update(string.punctuation, words_to_remove, stopwords_to_remove,
                  ge_areas, year, ge_prefixes)


def preprocess(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '', text_input)
    return output.lower().strip()


def clean_text(text):
    # add spaces and replace leading "and" or "&"
    return re.sub('^(and|&)', '', text.replace('\xa0', " ")).strip()


def remove_stopwords(text):
    filtered_words = [word.lower()
                      for word in text.split() if word.lower() not in filter_set]
    return " ".join(filtered_words)


def build_word_course_dict():
    df = pd.read_csv(course_scraper.FILE_NAME)
    df[DESCRIPTION] = df[DESCRIPTION].map(preprocess)
    df[DESCRIPTION] = df[DESCRIPTION].map(remove_stopwords)

    word_course_dict = collections.defaultdict(list)

    for index, row in df.iterrows():
        description = row[DESCRIPTION]
        prefix = row[COURSE_PREFIX]

        prefixes = format_course_prefixes(prefix)

        for word in description.split(' '):
            word_course_dict[word] += prefixes

    return word_course_dict


def format_course_prefixes(prefix_str: str):
    # Returns a list of course prefixes
    formatted_prefixes = []

    # one course listed
    if "/" not in prefix_str:
        formatted_prefixes.append(prefix_str.replace(" ", "-"))
        return formatted_prefixes

    # multiple courses
    split_prefixes = re.split('/| ', prefix_str)

    # crosslisted courses with different numbers (HNRS 304/ISLA 303)
    if (len(split_prefixes) % 2 == 0):
        prefixes = (prefix_str.split("/"))
        for prefix in prefixes:
            formatted_prefixes.append(prefix.replace(" ", "-"))
        return formatted_prefixes
    else:
        # crosslisted courses with different depts, same number (HIST/HNRS 335)
        prefixes, course_num = prefix_str.split(" ")
        prefixes = prefixes.split("/")
        for prefix in prefixes:
            formatted_prefixes.append(f'{prefix}-{course_num}')
    return formatted_prefixes


def build_course_program_dict():
    df = pd.read_csv("program_courses.csv")
    df["Program"] = df["Program"].map(clean_text)

    program_course_dict = collections.defaultdict(list)
    for index, row in df.iterrows():
        program = row["Program"]
        course_prefix = row["Course Prefix"]

        # handle mulitple prefixes EX. CPE/CSC 123
        prefixes = format_course_prefixes(course_prefix)
        # print(program, course_prefix, prefixes)
        for prefix in prefixes:
            program_course_dict[prefix] += [program]

    return program_course_dict


if __name__ == "__main__":
    d = build_word_course_dict()
    program_course_dict = build_course_program_dict()
    print(d)
    # print(program_course_dict)
