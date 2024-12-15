from ST_model import JointSpeechTranslationModel
import torch
from tqdm import tqdm
import os
from transformers import Wav2Vec2Processor, MBart50TokenizerFast
from datasets import load_metric, Dataset
from utils import set_seed
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import json

def prepare_dataset(split = "Training"):
    data_dir = f"./data/{split}/array/"
    text_dir = f"./data/{split}/label/"
    data = []
    for file_name in tqdm(os.listdir(data_dir)):
        path = os.path.join(data_dir, file_name)
        text_path = os.path.join(text_dir, file_name.replace(".npy", ".json"))
        with open(text_path, 'r') as f:
            t = json.load(f)
        text = t['MT']
        text_PE = t['MTPE']
        if text_PE=="":
            continue
        data.append({"path": path, "translation" : text_PE})
    return data

def collate_fn(batch, processor, tokenizer):
    input_values = [processor(np.load(sample["path"]), sampling_rate=16000, return_tensors="pt").input_values[0] for sample in batch]
    labels = [sample["translation"] for sample in batch]
    
    input_values = nn.utils.rnn.pad_sequence(input_values, batch_first=True)  # Padding input_values
    labels_batch = tokenizer(
        labels,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"]
    labels_batch[labels_batch == tokenizer.pad_token_id] = -100  # Loss 계산을 위한 padding 처리

    return {"input_values": input_values, "labels": labels_batch}



def compute_bleu(predictions, references, tokenizer, metric):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(references, skip_special_tokens=True)
    bleu_score = metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs])
    return bleu_score["score"]

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # Ignore padding tokens
        mask = target != self.ignore_index
        target = target[mask]
        logits = logits[mask]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
# 학습 루프 정의
def train(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    optimizer, 
    scheduler, 
    tokenizer, 
    device, 
    num_epochs=5, 
    log_interval=10, 
    save_path="./best_model.pt"
):
    bleu_metric = load_metric("sacrebleu", trust_remote_code=True)
    model.to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.3)
    best_bleu = -1  
    def clean_labels(labels):
        labels = labels.cpu().numpy()
        return [[token for token in label if token != -100] for label in labels]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values=input_values, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        val_bleu = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_values=input_values, labels=labels)
                val_loss += criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)).item()

                generated_ids = model.generate(input_values, max_length=128, num_beams=5)
                val_bleu += compute_bleu(generated_ids, clean_labels(labels), tokenizer, bleu_metric)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_bleu = val_bleu / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, BLEU: {avg_val_bleu:.2f}")

        if avg_val_bleu > best_bleu:
            best_bleu = avg_val_bleu
            print(f"New best BLEU score: {best_bleu:.2f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(save_path, f"{best_bleu:.2f}_model.pt"))

        scheduler.step()

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(save_path, f"{best_bleu:.2f}_model.pt"))) 
    model.eval()
    test_bleu = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(input_values, max_length=128, num_beams=5)
            test_bleu += compute_bleu(generated_ids, clean_labels(labels), tokenizer, bleu_metric)

    avg_test_bleu = test_bleu / len(test_loader)
    print(f"Test BLEU score: {avg_test_bleu:.2f}")


if __name__ == "__main__":
    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
    model = JointSpeechTranslationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for l in model.mbart_model.model.encoder.layers:
        l.fc1.requires_grad = False
        l.fc2.requires_grad = False
    for l in model.mbart_model.model.decoder.layers:
        l.fc1.requires_grad = False
        l.fc2.requires_grad = False
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")

    train_dataset = prepare_dataset(split="Training")
    test_dataset = prepare_dataset(split="Validation")
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

    train_dataset = Dataset.from_list(train_dataset)
    val_dataset = Dataset.from_list(val_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor, tokenizer),
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, processor, tokenizer),
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, processor, tokenizer),
        num_workers=4,
    )
    print("data statistics")
    print("train : ", len(train_dataset))
    print("val : ", len(val_dataset))
    print("test : ", len(test_dataset))
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        tokenizer,
        device,
        num_epochs=5,
        log_interval=10,
        save_path="./results/"
    )
