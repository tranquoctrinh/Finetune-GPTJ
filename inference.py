import pandas as pd
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, AutoConfig, AutoTokenizer, GPTJForCausalLM
from tqdm import tqdm
import torch
import os
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


CACHE_DIR = "cache/"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def inference(model, tokenizer, prompt, do_sample=True, num_beams=5, temperature=0.5, max_length=1024, flexible=True, num_return_sequences=5):
    model.eval()
    with torch.no_grad():
        texts = [tokenizer.special_tokens_map["bos_token"] + " " + prompt]
        encoding = tokenizer(texts, padding=False, return_tensors='pt').to(device)
        if encoding["input_ids"].shape[1] < 700:
            if do_sample:
                generated_ids = model.generate(
                    **encoding, 
                    do_sample=True,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                generated_ids = model.generate(
                    **encoding, 
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            lst_pred = []
            for i in range(num_return_sequences):
                pred = generated_texts[i]
                if len(pred.split("<language>")) == 2:
                    pred = "<language>" + pred.split("<language>")[-1]
                else:
                    pred = pred.split("<END>")[-1]
                pred = pred.replace("<b>", " ")
                pred = pred.replace("<language>", " <language> ").replace("<rewrite>", " <rewrite> ").replace("<title>", " <title> ").replace("<date>", " <date> ").replace("<photographer>", " <photographer> ").replace("<agency>", " <agency> ")
                pred = " ".join(pred.strip().split())
                lst_pred.append(pred)
            del encoding
            torch.cuda.empty_cache()
        else:
            lst_pred = ["SO LONG" for i in range(num_return_sequences)]
    return lst_pred


def main(df, task, pretrained_path):
    config = AutoConfig.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = GPTJForCausalLM(config).from_pretrained(pretrained_path)

    model.half()
    model = model.to(device)
    model.eval()

    generate = []
    for i,r in tqdm(df.iterrows(), total=df.shape[0], desc=f"Predict Task {task}"):
        fpath = os.path.join(CACHE_DIR, f"{r['pid']}_{task}.json")
        if os.path.exists(fpath):
            res = json.load(open(fpath))
            generate.append(res[task])
        else:
            pr = r["prompt"].replace("\n", " ").strip()
            pred = inference(
                model=model,
                tokenizer=tokenizer,
                prompt=pr,
                do_sample=True,
                num_return_sequences=5)
            print(f"---PROMPT:     {pr}")
            print(f"---COMPLETION: {r['compl']}")
            for ip, p in enumerate(pred):
                print(f"---CLEAN {ip}:    {p}")
                break
            print("-"*100)
            generate.append(pred)
            res = dict(r)
            res[task] = generate[-1]
            json.dump(res, open(fpath, "w"))
    return generate
    
if __name__ == "__main__":
    df = pd.read_parquet("/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/data/multi_task_train_test_validate_30dec.parquet")
    df = df[df["train_test_validate"]=="test"].reset_index(drop=True)
    def process_input(prompt):
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("<0>", "<language>").replace("<1>", "<envelope>").replace("<2>", "<back>").replace("<3>", "<front>")
        prompt = " ".join(prompt.split())
        prompt = prompt.strip()
        return prompt

    def process_output(completion):
        completion = completion.replace("\n", " ")
        completion = completion.replace("<0>", "<language>").replace("<5>", "<rewrite>").replace("<6>", "<title>").replace("<7>", "<date>").replace("<8>", "<photographer>").replace("<9>", "<agency>")
        completion = " ".join(completion.split())
        completion = completion.strip()
        return completion

    df["prompt"] = df["prompt"].apply(process_input)
    df["compl"] = df["compl"].apply(process_output)

    print(df.head())
    print(df.shape)
    pretrained_path = '/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/finetune_gptj6b'

    for task in ["multi-task_1jan"]:
        generate = main(df, task, pretrained_path)
        df[task] = generate
    df.to_csv(f"1jan_multi-task_gptj_lastest_transformers.csv", index=False)