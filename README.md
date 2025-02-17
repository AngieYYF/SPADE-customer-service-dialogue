# SPADE: Structured Prompting Augmentation for Dialogue Enhancement in Machine-Generated Text Detection

[**Paper**]() | [**Datasets**](#datasets) | [**Detection**](#detection) | [**Example**](#example) | [**Frameworks**](#frameworks)

## Introduction
**SPADE** contains:
- A repository of customer service line synthetic user dialogues with goals.
- A collection of implementations of synthetic user detection methods.
- An example of detecting synthetic user dialogues within the dataset.
- A collection of our proposed data augmentation frameworks.

## Datasets

You can download SPADE on HuggingFace.
Here is the link: https://huggingface.co/datasets/AngieYYF/SPADE-customer-service-dialogue

SPADE contains a number of datasets, including:

- Bona Fide (cleaned MultiWOZ 2.1 labelled by ConvLab-3): papers ([MultiWOZ 2.1](https://aclanthology.org/2020.lrec-1.53/), [ConvLab-3](https://arxiv.org/abs/2211.17148)), [source](https://github.com/ConvLab/ConvLab-3/tree/master/data/unified_datasets/multiwoz21), [file](dataset/cleaned_hotel_goal_dia.csv)
- Missing Sentence Completion: [gpt3.5](dataset/Missing_Sentence_gpt.csv), [llama](dataset/Missing_Sentence_llama.csv)
- Next Response Generation: [gpt3.5](dataset/Next_Response_gpt.csv), [llama](dataset/Next_Response_llama.csv)
- Goal to Dialogue: [gpt3.5](dataset/G2D_gpt.csv), [llama](dataset/G2D_llama.csv)
- Paraphrase Dialogue (synthetic system): [gpt3.5](dataset/Par_chatbot_system_gpt.csv), [llama](dataset/Par_chatbot_system_llama.csv)
- Paraphrase Dialogue (synthetic system and user): [gpt3.5](dataset/Par_full_chatbot_gpt.csv), [llama](dataset/Par_full_chatbot_llama.csv)
- End-to-End Conversation: [gpt3.5 system gpt3.5 user](dataset/E2E_Convo_gpt_gpt.csv), [gpt3.5 system llama user](dataset/E2E_Convo_gpt_llama.csv), [llama system llama user](dataset/E2E_Convo_llama_llama.csv), [llama system gpt3.5 user](dataset/E2E_Convo_llama_gpt.csv)

The datasets are of csv file format and contain the following columns:
| Dataset                    | Column            | Description                                                                                              |
|----------------------------|-------------------|----------------------------------------------------------------------------------------------------------|
| **All**                     | *dia_no* / *new_dia_no* | Unique ID for each dialogue. Dialogues with the same ID across datasets are based on the bona fide dialogue with the same *new_dia_no*. |
|                            | *dia*             | The dialogue itself, either bona fide or synthetic.                                                      |
| **Bona Fide**               | *new_goal*        | The cleaned user goal associated with the dialogue.                                                      |
| **Next Response Generation**| *turn_no*         | Zero-based turn number of the user response within the dialogue.                                         |
|                            | *context*         | Context provided to the user for generating the next response.                                           |
|                            | *response*        | Single utterance response generated by the user based on the provided context.                           |




## Detection
The detection folder contains implementations of detection methods of three classes: 
- [feature](detection/feature): extraction of features and implementation of MLP classifier.
- [pretrained language model](detection/roberta): implementation of RoBERTa-based classifier.
- [statistical](detection/statistical): extraction of statistical features.

## Example

An example of using the provided datasets and performing synthetic user detection is available [here](detection/roberta_detection.ipynb).

## Framworks

Examples of prompt for different data augmentation frameworks are available [here](frameworks/framework_table.ipynb).


## License
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
