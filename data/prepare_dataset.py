from datasets import load_dataset, DatasetDict
from morphpiece import MorphPiece

block_size=1024

raw_datasets = load_dataset('parquet',data_dir="/pfss/mlde/workspaces/mlde_wsp_MorphPiece/data/fineweb-edu/sample/100BT",split='train').select_columns(['id','text'])

raw_datasets = raw_datasets.select(range(int(0.6*97270686))) # 60% of 97270686

raw_datasets = DatasetDict(raw_datasets.train_test_split(test_size=0.001))

tokenizer = MorphPiece(data_dir='/pfss/mlde/workspaces/mlde_wsp_MorphPiece/data/tokenizers_50k/gpt2-owt-morph15k')

text_column_name='text'
def tokenize_function(examples,tokenizer):
    return tokenizer(examples[text_column_name],return_attention_mask=False)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = raw_datasets['train'].map(
            tokenize_function, 
            fn_kwargs={'tokenizer':tokenizer},
            batched=True,
            num_proc=16,
            remove_columns=raw_datasets['train'].column_names,
        )


lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=16,
        )

lm_datasets.save_to_disk(f'/pfss/mlde/workspaces/mlde_wsp_MorphPiece/data/fineweb-edu/grouped_owt_morph15k_50k_60B_{block_size}')