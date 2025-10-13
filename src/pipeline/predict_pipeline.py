import sys
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def _patch_onehot_encoder(obj):
    """Recursively find OneHotEncoder instances in pipelines/columntransformers and
    ensure the attribute `_drop_idx_after_grouping` exists to maintain compatibility
    when unpickling objects saved with older sklearn versions.
    This is a low-risk compatibility shim for inference only.
    """
    try:
        if isinstance(obj, OneHotEncoder):
            # attribute introduced/used in some sklearn versions; ensure it's present
            if not hasattr(obj, "_drop_idx_after_grouping"):
                setattr(obj, "_drop_idx_after_grouping", None)

        # ColumnTransformer: iterate its transformers_
        elif isinstance(obj, ColumnTransformer):
            for name, trans, cols in getattr(obj, 'transformers_', []):
                _patch_onehot_encoder(trans)

        # Pipeline: iterate steps
        elif isinstance(obj, Pipeline):
            for name, step in getattr(obj, 'steps', []):
                _patch_onehot_encoder(step)

    except Exception:
        # don't let the shim raise; we only attempt best-effort patching
        pass
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # Attempt to patch deserialized preprocessor for cross-version OneHotEncoder
            _patch_onehot_encoder(preprocessor)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

