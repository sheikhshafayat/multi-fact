import json
from tqdm import tqdm
import random
import os
import argparse

from openai import OpenAI
openai_api_key = open("openai-api.txt").read().strip()
client = OpenAI(api_key=openai_api_key)

def get_bio_gpt(name, language, model="gpt-3.5-turbo-0613"): 
    
    # set prompts for each language
    if language == "English":
        prompt = f"Write a biography of {name}"
    elif language == "Bengali":
        prompt = f"{name} এর জীবনী লেখ"
    elif language == "Korean":
        prompt = f"{name}에 대한 전기를 작성하시오"
    elif language == "Russian":
        prompt = f"Напиши биографию {name}"
    elif language == "Chinese":
        prompt = f"编写{name}的传记"
    elif language == "Arabic":
        prompt = f"اكتب سيرة ذاتية لـ{name}"
    elif language == "French":
        prompt = f"Écrivez une biographie de {name}"
    elif language == "Spanish":
        prompt = f"Escribe una biografía de {name}"
    elif language == "German":
        prompt = f"Schreibe eine Biografie von {name}"
    elif language == "Japanese":
        prompt = f"{name}の伝記を書く"
    elif language == "Swahili":
        prompt = f"Andika wasifu wa {name}"

    
    response = None 
    while response is None:
        response = client.chat.completions.create(
        model=model,
        seed=42,
        messages=[
            {"role": "user", "content": prompt}
        ]
        )
    bio = response.choices[0].message.content
    return bio

if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="language", default="Bengali")
    parser.add_argument("--model", type=str, help="model", default="gpt-3.5-turbo-0613")

    args = parser.parse_args()
    print("Starting biography generation...")
    language = args.language
    print(f"Language: {language} using model {args.model} to generate")
    assert language in ["English", "Bengali", "Korean", "Russian", "Chinese", "Arabic", "French", "Spanish", "German", "Japanese", "Swahili"]; "Invalid language"

    # get people list in corresponding language
    with open("necessary_dicts/lang_name_dict.json") as f:
        lang_name_dict = json.load(f)
    
    people_list = lang_name_dict[language]

    # generate biographies
    bios = [{name:get_bio_gpt(name, language, args.model) for name in tqdm(people_list)}]

    # save bios
    if args.model == "gpt-3.5-turbo-0613":
        _model = "gpt3.5"
    elif args.model == "gpt-4-turbo-0613":
        _model = "gpt4"
    else:
        _model = args.model
    
    with open(f"generated/presidents/{_model}/{language}_orig_bios.json", "w") as f:
        json.dump(bios, f)