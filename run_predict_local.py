from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pandas as pd
import traceback

# Create the same input
try:
    data=CustomData(
        gender='male',
        race_ethnicity='group B',
        parental_level_of_education='some college',
        lunch='standard',
        test_preparation_course='none',
        reading_score=17.0,
        writing_score=1.0
    )
    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print('Before Prediction')
    pp=PredictPipeline()
    print('Mid Prediction')
    res=pp.predict(pred_df)
    print('Result:', res)
except Exception as e:
    print('EXCEPTION OCCURRED:')
    traceback.print_exc()
