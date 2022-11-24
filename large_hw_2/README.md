# Primary Home Assignment 2: Machine Translation

In this assignment, your goal will be to build a pipeline for machine translation and implement several decoding schemes.

## Data 
We will use the [IWSLT 2017 German to English](https://wit3.fbk.eu/2017-01-b) translation dataset for all of 
our experiments. 
You can download the dataset with `gdown --fuzzy "https://drive.google.com/file/d/12ycYSzLIG253AFN35Y6qoyf9wtkOjakp/view?usp=sharing"`
(make sure to run `pip install gdown` beforehand).

The files we are going to use are:
* Training data (`train.tags.de-en.de`, `train.tags.de-en.en`),  
* Validation data (`IWSLT17.TED.dev2010.de-en.de.xml`, `IWSLT17.TED.dev2010.de-en.en.xml`)
* Test data (`IWSLT17.TED.tst2010.de-en.de.xml`, `IWSLT17.TED.tst2010.de-en.en.xml`)

The format description of these files is contained in README inside the archive. 
For training data, you need all sentences inside lines not enclosed in angular brackets.
For evaluation data, all sentences are enclosed within <seg id="..."> blocks.

## Tasks

This assignment has 6 compulsory tasks and two bonus ones:

### Task 1 (1 point)
Using the [datasets](https://github.com/huggingface/datasets) library, parse the dataset files into 
[Dataset](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset) objects.
Implement the following functions inside data.py: `process_training_file`, `process_evaluation_file`, `convert_files`, `TranslationDataset`.

### Task 2 (0.5 points)
Using the [tokenizers](https://github.com/huggingface/tokenizers) library, build a Byte-Pair Encoding vocabulary from training and validation data.
The size of the vocabulary should be 30,000 tokens for both languages.

### Task 3 (1.5 points)
Implement the translation model using the [torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer) class.

### Task 4 (2 points)
Train this model on the tokenized training dataset from above with **at most 50 epochs** over all examples.
The model **must not have more than 500 million parameters** in total; you are free to choose and tune other hyperparameters as you wish.

Once the model is trained, get the test set predictions with **greedy decoding** and measure the translation quality 
with `sacrebleu==2.3.1` using its Python API. You should get a BLEU score of at least 10 if everything is correct.

### Task 4 (2 points)
Implement the **beam search** algorithm for decoding (see [decoding.py](./decoding.py) for a template). 
Obtain and score the predictions of your model with the beam size of 5.

### Task 5 (2 points)
Optimize beam search by implementing its **batched** version: instead of processing one example 
at a time, it should process a batch of examples in parallel. Note that the result of this decoding should be 
equivalent to the result of running code from Task 4: you can check this by running the decoding with a batch size of 1.
You can reuse the template of the beam search function to implement it.

### Task 6 (1 points)
Implement a [**constrained**](https://aclanthology.org/N18-1119/) version of batched beam search that allows to force a list of strings 
given by the user to appear in the translation. 
Validate that it works manually by creating a script that takes a pretrained model and several "hints" and applies it to the test set.

### Task 7 (bonus, up to 2 points): 
Implement a model for *non-autoregressive* translation using Mask-Predict: this approach allows speeding up generation by predicting all tokens at the same time and then refining the prediction. You can use the method from [the paper](https://arxiv.org/abs/1904.09324) and implement both training and inference using the templates given earlier.

### Task 8 (bonus, up to 2 points):
Implement a model for *simultaneous* translation: this approach allows to generate the prediction as soon as parts of the input are received, which is very helpful for live speech translation. Since the encoder usually attends to the entire input sequence, we need to constrain attention to be *monotonic*: tokens from the output can only attend to some prefix of the input sequence, and the prefix size is non-decreasing with the length of the prediction. You can use the method from [this paper](https://arxiv.org/abs/1704.00784)  and implement both training and inference using the templates given earlier.

Both bonus assignments require that you submit **a report** along with your assignment: it can be a PDF file, a Jupyter notebook, or a Weights and Biases report that details your solution, difficulties that you faced when implementing it, and shows some examples (kudos for visualizations!).

## Deadline
There are two deadlines for the assignment:
* **Intermediate deadline (December 6, 08:00MSK):** you need to submit the pipeline that reads the data, builds 
a tokenizer and trains a model that achieves at least 5 BLEU on the test set.
* **Final deadline (December 11, 08:00MSK):** you need to submit the entire assignment.
All deadlines are strict.

Each submission needs to contain `data.py`, `model.py`, `process_data.py`, `train_model.py` and `decoding.py`. 
Also, upload the `trained` model to any cloud storage of your choice or as a W&B artifact and attach the link to your submission.
If you have completed a bonus task, please make sure to submit your report and corresponding code as well.

## Available packages 
The list of all packages explicitly available to you with specific versions is given in the [requirements.txt](./requirements.txt) file.
All additional libraries (excluding modules of the Python 3.10 standard library) are not guaranteed to be installed: 
if your code does not run on the system because of `ImportError`s, you will not get points for corresponding tasks.
However, if you wish to request adding a specific library to the requirements file, you may discuss it with the lecturer until the final deadline.

## Plagiarism
Sharing code of your solution with fellow students is prohibited.
If you have discussed any parts of the assignment with other students or used materials from PyTorch help/tutorials, 
make sure to state this in Anytask when submitting the assignment.
Copying code from any source other than PyTorch help or tutorials is not allowed.