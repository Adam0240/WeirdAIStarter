# Weird AI Starter Repository

Weird AI is a parody lyric generation project for an AI Engineering course.
You should see the badges below move from failing to passing as you work through developing the model. 

| Lesson                                    | Status                                                                      |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| Lesson 02 - Tokenization                  | ![Lesson 02](../../actions/workflows/lesson-02-tests.yml/badge.svg)         |
| Lesson 03 - Attention                     | ![Lesson 03](../../actions/workflows/lesson-03-tests.yml/badge.svg)         |
| Lesson 04 - GPT Model                     | ![Lesson 04](../../actions/workflows/lesson-04-tests.yml/badge.svg)         |
| Lesson 05 - Unlabeled Data                | ![Lesson 05](../../actions/workflows/lesson-05-tests.yml/badge.svg)         |
| Lesson 06 - Classification                | ![Lesson 06](../../actions/workflows/lesson-06-tests.yml/badge.svg)         |
| Lesson 07 - Generation                    | ![Lesson 07](../../actions/workflows/lesson-07-tests.yml/badge.svg)         |
| Lesson 08 - Understanding Reasoning       | ![Lesson 08](../../actions/workflows/lesson-08-tests.yml/badge.svg)         |
| Lesson 09 - Pretrained Generation         | ![Lesson 09](../../actions/workflows/lesson-09-tests.yml/badge.svg)         |
| Lesson 10 - Eval. Reasoning               | ![Lesson 10](../../actions/workflows/lesson-10-tests.yml/badge.svg)         |
| Lesson 11 - Infer. Time Scaling           | ![Lesson 11](../../actions/workflows/lesson-11-tests.yml/badge.svg)         |
| Lesson 12 - Self Refinement               | ![Lesson 12](../../actions/workflows/lesson-12-tests.yml/badge.svg)         |
| Lesson 13 - Distillation                  | ![Lesson 13](../../actions/workflows/lesson-13-tests.yml/badge.svg)         |
| Lesson 14 - Reinforcement Learning        | ![Lesson 14](../../actions/workflows/lesson-14-tests.yml/badge.svg)         |
| Lesson 15 - Improving GRPO                | ![Lesson 15](../../actions/workflows/lesson-15-tests.yml/badge.svg)         |
| Final Project - Lightweight Reinforcement | ![Final Project](../../actions/workflows/final-project-tests.yml/badge.svg) |

## Project Goals

By the end of the course, this project will support:

- Collecting, cleaning, and organizing song lyrics so they can be used as training data for an AI model
- Tokenization of text (encoding and decoding)
- Transformer-based text generation
- Instruction-following parody generation
- Rhyme and syllable evaluation
- Self-refinement
- Reasoning-enhanced generation
- Reinforcement training
- Saving and deploying a model

## Tasks

The tasks needed to build the Weird AI model from scratch are outlined in the *assignments* folder.
The assignments follow the two books used when teaching the in-person class.
- 1 - 7 follow the book Building a Large Language Model (From Scratch)
- 8 - 15 follow the book Building a Reasoning Model (From Scratch) with the following caveat:
  - Assignment 10 corresponds with the chapter on *Evaluating Reasoning Models*
    - In that chapter, the author uses mathematical analysis to test chain of thought and other reasoning abilities.  This works well in the demonstration code because a math problem has a clear right answer
    - Since there is not a clearcut way to measure song parodies, a different approach to evaluation is used in this project
  - Assignment 13 corresponds with Chapter 8, Distilling reasoning models is covered as lesson 13
  - Assignments 14 and 15 correspond with Chapters 6 and 7
- FinalProject.md goes beyond these two books and completes the Weird AI project
  - Implementing a Mini-GRPO for rewards-based training of Weird AI
  - Building a deployment of the model
  - Loading the deployable model into a web application using gradio and interacting with the model through this interface

## Project Setup

The project is configured to leverage VS Code as the developer environment.  It is recommended that you fork the repository and periodically synchronize your fork to get any updates to the project and codebase.

### Create a virtual environment:

```bash
python -m venv .venv
```

### Then activate the environment:
Windows:
```PowerShell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Run tests

Unit tests can be executed with the following command:
```bash
python -m pytest tests/<test_file_to_run>.py
```

## Prepare Data
```bash
python scripts/prepare_data.py
```

## Train Model

```bash
python scripts/train_model.py
```

## Launch Demo/Generate Lyrics
```bash
python app/gradio_app.py
```

After the model is trained and saved in a format it can be restored in, the gradio application will work. 
- This happens in the final assignment
