import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


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

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.add_tokens(["<language>","<envelope>","<back>","<front>","<rewrite>","<title>","<date>","<photographer>","<agency>", "<END>", "<b>"])

df = pd.read_parquet("/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/data/multi_task_train_test_validate.parquet")
df["prompt"] = df["prompt"].apply(process_input)
df["compl"] = df["compl"].apply(process_output)

dftrain = df[df["train_test_validate"]=="train"].reset_index(drop=True)
dftrain["text"] = ["<|endoftext|> " + pr + " "+ comp + " <|endoftext|>" for pr,comp in dftrain[["prompt", "compl"]].values]
dftrain["length"] = [len(x) for x in dftrain["text"].values]
dftrain["length_token"] = [len(tokenizer(x)["input_ids"]) for x in tqdm(dftrain.text.values)]
dftrain[["text"]].to_csv("/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/data/train.csv", index=False)


# dftest = df[df["train_test_validate"]=="test"].reset_index(drop=True)
# dftest["text"] = ["<|endoftext|> " + pr + " "+ comp + " <|endoftext|>" for pr,comp in dftest[["prompt", "compl"]].values]
# dftest["length"] = [len(x) for x in dftest["text"].values]
# dftest[["text"]].to_csv("/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/data/test.csv", index=False)


dftest = df[df["train_test_validate"]=="validate"].reset_index(drop=True)
dftest["text"] = ["<|endoftext|> " + pr + " "+ comp + " <|endoftext|>" for pr,comp in dftest[["prompt", "compl"]].values]
dftest["length"] = [len(x) for x in dftest["text"].values]
dftest[["text"]].to_csv("/home/ims_dev_ml/finetune_gptj6b/huggingface/multi_task/data/val.csv", index=False)