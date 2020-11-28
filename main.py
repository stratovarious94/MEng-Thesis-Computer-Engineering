import sys

from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.generators import ImageGenerator
from src.machine_learning.train import Trainer
from src.machine_learning.predict import Predictor
from src.evaluating.evaluate import Evaluator
from src.evaluating.plot import Plotter


# python main.py preprocess

# python main.py train vgg16
# python main.py predict vgg16
# python main.py evaluate vgg16
# python main.py pipeline vgg16

# python main.py train efficientnet
# python main.py predict efficientnet
# python main.py evaluate efficientnet
# python main.py pipeline efficientnet

# python main.py train vit
# python main.py predict vit
# python main.py evaluate vit
# python main.py pipeline vit
action, model_name = sys.argv[1:3]
if action == 'preprocess':
    Preprocessor().preprocessing_pipeline()
elif action == 'train':
    img_gen = ImageGenerator()
    Trainer(img_gen.get_train_generator(), img_gen.get_valid_generator(), model_name).train()
elif action == 'predict':
    img_gen = ImageGenerator()
    Predictor(img_gen.get_test_generator(), model_name).predict()
elif action == 'evaluate':
    Evaluator(model_name).evaluate()
elif action == 'plot':
    Plotter(model_name).plot()
elif action == 'pipeline':
    img_gen = ImageGenerator()
    Trainer(img_gen.get_train_generator(), img_gen.get_valid_generator(), model_name).train()
    Predictor(img_gen.get_test_generator(), model_name).predict()
    Evaluator(model_name).evaluate()
