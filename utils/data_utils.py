import os
import json
import hashlib
import datasets
import pandas as pd


def extract_data_from_qa_json(file_path, seed):

    dataset_dict = {
            "question": [],
            "answers": [],
            'context': [],
            "id": []
        }
    question_id = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f) 

    # Extract required fields and populate dataset_dict
    for row in data:
        # print("Question:", row["Question"], "Answer:", row["Answer"], "Knowledge:", row["Knowledge"])
        dataset_dict["question"].append(row["Question"])
        dataset_dict["answers"].append({"text": [row["Answer"]], "answer_start": [0]})  # answer_start is needed for QA models
        dataset_dict["context"].append(row["Knowledge"])
        dataset_dict["id"].append(str(question_id))  # Convert index to string for consistency
        question_id += 1

    dataset = datasets.Dataset.from_dict(dataset_dict)

    return dataset

def extract_data_from_yes_no_json(file_path, seed):

    dataset_dict = {
            "question": [],
            "answers": [],
            "context": [],
            "ref_answer": [],
            "id": []
        }
    question_id = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f) 

    # Extract required fields and populate dataset_dict
    for row in data:
        dataset_dict["question"].append(row["Question"])
        # whether the answer is supported by the context
        hallucination_label = row["Hallucination"]
        if hallucination_label == 0:
            answer = "Yes"
        elif hallucination_label == 1:
            answer = "No"
        dataset_dict["answers"].append({"text": [answer], "answer_start": [0]})  # answer_start is needed for QA models
        dataset_dict["context"].append(row["Knowledge"])
        dataset_dict["ref_answer"].append(row["Answer"])
        dataset_dict["id"].append(str(question_id))  # Convert index to string for consistency
        question_id += 1

    # dataset = datasets.Dataset.from_dict(dataset_dict).shuffle(seed)
    dataset = datasets.Dataset.from_dict(dataset_dict)

    return dataset

def load_ds(dataset_name, seed, data_path):
    """Load dataset."""

    train_dataset, validation_dataset = None, None

    if dataset_name == "bioasq":
        # use the provided Hydra-resolved path
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_dict = {
            "question": [],
            "answers": [],
            "context": [],
            "id": []
        }
        for i, row in enumerate(data):
            dataset_dict["question"].append(row["question"])
            dataset_dict["answers"].append({"text": [row["answer"]], "answer_start": [0]})
            dataset_dict["context"].append(row["context"])
            dataset_dict["id"].append(str(i))

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif dataset_name == "halueval":
        # data_path is already passed in from Hydra, so we just use it
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_dict = {
            "question": [],
            "answers": [],
            "context": [],
            "id": []
        }

        for i, row in enumerate(data):
            dataset_dict["question"].append(row["question"])
            dataset_dict["answers"].append(
                {"text": [row["answer"]], "answer_start": [0]}
            )
            dataset_dict["context"].append(row["context"])
            dataset_dict["id"].append(str(i))

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)

        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "esghalu":
        base = data_path      
        train_path = os.path.join(base, "esghalu_train.json")
        test_path  = os.path.join(base, "esghalu_test.json")

        train_dataset = extract_data_from_qa_json(train_path, seed)
        validation_dataset = extract_data_from_qa_json(test_path, seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, validation_dataset

