import time
import traceback
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object

features = pd.DataFrame({'gender':['male'],'race_ethnicity':['group B'],'parental_level_of_education':['some college'],'lunch':['standard'],'test_preparation_course':['none'],'reading_score':[17.0],'writing_score':[1.0]})

print('Start test script')
pp = PredictPipeline()

try:
    print('About to call predict()')
    start = time.time()
    preds = pp.predict(features)
    duration = time.time() - start
    print('predict() returned in', duration, 'seconds. preds =', preds)
except Exception as e:
    print('predict() raised an exception:')
    traceback.print_exc()

# Now load model and preprocessor directly to test steps separately
try:
    print('\nDirect load objects test')
    model_path = 'artifacts/model.pkl'
    preprocessor_path = 'artifacts/preprocessor.pkl'
    print('Loading model...')
    mstart = time.time()
    model = load_object(file_path=model_path)
    mload = time.time() - mstart
    print('Model loaded in', mload, 's; type:', type(model))

    print('Loading preprocessor...')
    pstart = time.time()
    preprocessor = load_object(file_path=preprocessor_path)
    pload = time.time() - pstart
    print('Preprocessor loaded in', pload, 's; type:', type(preprocessor))

    print('Calling preprocessor.transform')
    tstart = time.time()
    data_scaled = preprocessor.transform(features)
    ttime = time.time() - tstart
    print('transform completed in', ttime, 's; shape:', getattr(data_scaled, 'shape', 'N/A'))

    print('Calling model.predict')
    pstart = time.time()
    preds2 = model.predict(data_scaled)
    ptime = time.time() - pstart
    print('model.predict completed in', ptime, 's; preds:', preds2)

except Exception as e:
    print('Direct load/predict raised:')
    traceback.print_exc()

print('Test script finished')
