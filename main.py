import sys

from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.generators import ImageGenerator
from src.machine_learning.train import Trainer
from src.machine_learning.predict import Predictor
from src.evaluating.evaluate import Evaluator
from src.evaluating.plot import Plotter
from src.evaluating.explainable_convnets import Explainer

"""
Interface for the available actions that can be performed in the available models

Command generic type: python main.py <action> <model>

action: [preprocess, train, predict, evaluate, plot, explain, pipeline]
model: [vgg16, resnet, densenet, mobilenet, efficientnet, vit]

pipeline option calls the whole core pipeline for the model (train, predict, evaluate)
"""

action, model_name = sys.argv[1:3]
if action == 'preprocess':
    Preprocessor().preprocessing_pipeline()
elif action == 'train':
    img_gen = ImageGenerator()
    Trainer(img_gen.get_train_generator(), img_gen.get_valid_generator(), img_gen.get_test_generator(), model_name).train()
elif action == 'predict':
    img_gen = ImageGenerator()
    Predictor(img_gen.get_test_generator(), model_name).predict()
elif action == 'evaluate':
    Evaluator(model_name).evaluate()
elif action == 'plot':
    Plotter(model_name).plot()
elif action == 'explain':
    Explainer(model_name).explain()
elif action == 'pipeline':
    img_gen = ImageGenerator()
    Trainer(img_gen.get_train_generator(), img_gen.get_valid_generator(), model_name).train()
    Predictor(img_gen.get_test_generator(), model_name).predict()
    Evaluator(model_name).evaluate()
