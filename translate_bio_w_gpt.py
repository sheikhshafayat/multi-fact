import json
from tqdm import tqdm
import retry
import argparse
from openai import OpenAI


openai_api_key = open("openai-api.txt").read().strip()
client = OpenAI(api_key=openai_api_key)
class SomeError(Exception):
    pass

@retry.retry(SomeError, tries=3, delay=1)
def gpt_translate(text, language, model="gpt-3.5-turbo-0125"):
    response = None 
    system_prompt = f"""You are given a text in {language} and your job is to translate it into English.
    Do not make up any information, change anything, only translate the text. Do not add anything else. Only output the translation."""
    while response is None:
        response = client.chat.completions.create(
        model=model,
        seed=42,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{text}"}
        ]
        )
    translation = response.choices[0].message.content
    # translation = json.loads(translation)['name']
    return translation


if __name__ == "__main__":

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="language", default="Bengali")
    parser.add_argument("--model", type=str, help="model", default="gpt-3.5-turbo-0125")

    args = parser.parse_args()
    language = args.language

    assert language in ["Bengali", "Korean", "Russian", "Chinese", "Arabic", "French", "Spanish", "German", "Japanese", "Swahili"]; "Invalid target language"
    print(f"Starting translating from {args.language} to English...")


    # read json. change the path as needed
    with open(f"generated_bios/original/gpt4/{language}_orig_bios.json", "r") as f:
        bios = json.load(f)
    bios = list(bios[0].values())

    translated = []
    for bio in tqdm(bios):

        # tr_text = get_translation(bio, source_lang=language)
        tr_text = gpt_translate(bio, language)
        # replace all . with ". " 
        tr_text = tr_text.replace(".", ". ")
        translated.append(tr_text)
    
    with open("generated_bios/original/gpt3.5/English_orig_bios.json", "r") as json_file:
        en_bios = json.load(json_file)
    english_names = list(en_bios[0].keys())

    translated_bios = {name:bio for name, bio in zip(english_names, translated)}

    # change the path as needed
    with open(f"generated_bios/translated2en/gpt4-en-translated/{language}_2en_bios.json", "w") as f:
        json.dump(translated_bios, f)
    print(f"Translation to English from {args.language} complete")
