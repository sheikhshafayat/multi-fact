from factscore.factscorer import FactScorer
import numpy as np
import pandas as pd
import json, re, jsonlines
import argparse
from tqdm import tqdm
import yaml

def main(args):
    fs = FactScorer(openai_key=args.openai_key,
                data_dir=args.data_dir, # put your own cache directory where wikipedia is placed
                fact_generation_type=args.fact_generation_type, 
                model_name=args.model_name)

    
    print(f"Starting to calculate scores for {args.language}...")

    
    with open(f"generated_bios/translated2en/{args.generation_model}-en-translated/{args.language}_2en_bios.json", "r") as f:
        bios = json.load(f)
    if args.language == "English":
        bios = bios[0] # for English only; the data structure is slightly different because English hasn't been translated
    generations = list(bios.values())
    topics = list(bios.keys())

    score_dict = {}
    for topic, generation in tqdm(zip(topics, generations)):
        out = fs.get_score([topic], [generation], gamma=10)
        score_dict[topic] = out
        
    
    save_path = f"results/{args.generation_model}/{args.language}_2en_scores.json"

    with open(save_path, "w") as f:
        json.dump(score_dict, f)
       


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--openai_key", type=str, help="path to OpenAI API key", default="./openai-api.txt")
    parser.add_argument("--data_dir", type=str, help="path to cache directory", default="/mnt/nas2/sheikh/factscore_cache")
    parser.add_argument("--fact_generation_type", type=str, help="Mistral or InstrucGPT", default="Mistral")
    parser.add_argument("--model_name", type=str, help="retrieval+mistral+npm, retrieval+ChatGPT, retrieval+ChatGPT+npm, retrieval + mistral, npm", default="retrieval+mistral+npm")
    parser.add_argument("--language", type=str, help="language", default="English")
    parser.add_argument("--generation_model", type=str, help="gpt3.5 or gpt4", default="gpt4")

    args = parser.parse_args()

    language = args.language

    assert args.language in ["Bengali", "Korean", "Russian", "Chinese", "Arabic", "French", "Spanish", "German", "Japanese", "Swahili"]; "Invalid language"
    main()
    


