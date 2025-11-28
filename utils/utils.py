import logging
from utils.huggingface_models import HuggingfaceModel
from evaluate import load
import pickle

def is_answerable(generation):
    return len(generation['reference']['answers']['text']) > 0


def save(data, file, output_dir):
    with open(f'{output_dir}/{file}', 'wb') as f:
        pickle.dump(data, f)


# def init_model(model_name, max_new_tokens):
#     if any(x in model_name.lower() for x in ("llama", "mistral", "gemma")):
#         model = HuggingfaceModel(
#             model_name, stop_sequences='default',
#             max_new_tokens=max_new_tokens, token_layer=17)
#     else:
#         raise ValueError(f'Unknown model name `{model_name}`.')
#     return model
def init_model(model_cfg):
    return HuggingfaceModel(model_cfg)


def get_make_prompt(use_context=False):
    # def make_prompt(context, question, answer):
    def make_prompt(context, question, answer):
        prompt = ''
        prompt += f"Question: {question}\n"
        if use_context and (context is not None):
            prompt += f"[S_REPORT] {context} [E_REPORT]"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += 'Answer:'
        return prompt

    return make_prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
           set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference


def get_metric(metric):
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    elif metric == 'llm_gpt-3.5':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4o':
        metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric


def get_gpt_metric(metric_name):
    model_name = '_'.join(metric_name.split('_')[1:])

    class EntailmentGPT():
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    # prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    prompt = (
    "We are evaluating whether a proposed answer correctly conveys the expected meaning of an answer to the following question, based on the given context.\n\n"
    # f"Context: {example['context']}\n\n"
    f"Question: {example['question']}\n\n"
    )
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        # prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
        # prompt += "Does the proposed answer accurately convey the same meaning as the expected answer based on the context?"
        prompt += " Does predicted answer match with expected answer, beware expected answer may also contains the explanation of the the answer. "
    else:
        prompt += "Does the proposed answer accurately convey the same meaning as any of the expected answers based on the context?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')

        result = model.predict(prompt, 1)

        if isinstance(result, tuple):
            if len(result) == 3:
                predicted_answer, _, _ = result
            else:
                predicted_answer = result[0]  # Take the first item
        elif isinstance(result, str):  
            predicted_answer = result  # Directly use the string output
        else:
            raise ValueError(f"Unexpected output type from model.predict: {type(result)} -> {result}")

        # if len(model_pre) == 3:
        #     print("model.predict(prompt, 1):",model.predict(prompt, 1))
        #     predicted_answer, _, _ = model.predict(prompt, 1)
        # elif len(model_pre) == 1:
        #     predicted_answer = model.predict(prompt, 1)
        print('predicted_answer :',predicted_answer )
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)
