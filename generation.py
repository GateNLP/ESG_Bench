import logging
import random
import gc
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from utils import utils
from utils.data_utils import load_ds
import os


@hydra.main(version_base=None, config_path="conf", config_name="sentence_length_generation.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.experiment.seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(cfg)
    cfg.data.use_context = True

    # load dataset
    dataset_name=cfg.data.name
    train_dataset, validation_dataset = load_ds(dataset_name=cfg.data.name, seed=cfg.experiment.seed, data_path=cfg.data.datasets[dataset_name].path)
    logging.info('Train dataset: %s', train_dataset)
    logging.info('Validation dataset: %s', validation_dataset)

    if cfg.data.answerable_only:
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]
 
    make_prompt = utils.get_make_prompt(cfg.data.use_context)

    # Initialize the model
    model = utils.init_model(cfg.model)


    for dataset_split in ['validation']:
        logging.info('Starting with dataset_split %s.', dataset_split)
        # save all input data and model predictions
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        generation_df_lst = []
        y_true_ls = []
        y_pred_ls = []
        dataset = validation_dataset
        possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(cfg.num_eval_samples, len(dataset)))
        it = 0
        # for index in tqdm(indices):
        for index in tqdm(indices):
                full_responses = []
                if (it + 1 % 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                it += 1
                # Grab example at index.
                example = dataset[index]
                question, context = example["question"], example['context']
               
                generations[example['id']] = {'question': question, 'context': context}
                correct_answer = example['answers']['text']
                current_input = make_prompt(context, question, None) 

    
                temperature = cfg.model.temperature
                # predicted_answer, token_log_likelihoods, embedding = model.predict(
                predicted_answer, _, _ = model.predict(
                    current_input, temperature, brief_prompt=cfg.prompt.brief_prompt
                )

                most_likely_answer_dict = {
                    'response': predicted_answer}
                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example)})
                full_responses.append((predicted_answer))
                model_id = f"{cfg.model.family}_{cfg.model.dataset}_{cfg.model.variant}"
                generation_df_lst.append([example['id'], model_id, temperature, current_input, predicted_answer,
                                        correct_answer])

                # Append all predictions for this example to `generations`.
                generations[example['id']]['responses'] = full_responses
        
        generation_df_cols = ['id', 'model_name', 'temperature', 'prompt', 'pred_answer',
                                'reference_answer']
        generation_df = pd.DataFrame(generation_df_lst, columns=generation_df_cols)
        generation_df.to_csv(f'{output_dir}/validation_uncertainty_measures.csv', index=False)

        # Save generations for that split.
        logging.info('Saving generations for %s. into %s', dataset_split, output_dir)
    logging.info(f'Output saved to {output_dir}')
    logging.info('Run complete.')
    del model
    

if __name__ == "__main__":
    main()
