from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Download dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

train = ds['train']
validation = ds['validation']
test = ds['test']

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Remove whitespaces - ensures words like "it is" are not tokenized together
tokenizer.pre_tokenizer = Whitespace()

# Initialize trainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True)

tokenizer.train()