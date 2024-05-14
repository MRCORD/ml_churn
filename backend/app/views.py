from fastapi import APIRouter, Depends
from app.models import *
from app.utils import *
import logging

# Create an APIRouter instance
router = APIRouter()

# Create a logger for the views module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the log level to INFO

# Create a file handler and set its log level
file_handler = logging.FileHandler('app.log')  # Create or append to the 'app.log' file
file_handler.setLevel(logging.INFO)

# Define a formatter for file output
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


@router.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to your API!"}


@router.get("/v1/data")
def get_max_values(data = Depends(load_data)):
    try:
        max_values = compute_max_values(data)
        logger.info(f"Max values computed: {max_values}")
        # logger.info(f"Type of max_values: {type(max_values)}")
        return {"message": "Max values computed", "max_values": max_values}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}
    

@router.get("/v1/pickles_load")
def get_pickles():
    try:
        X_train = load_x_y("data/X_train.pkl")
        # logger.info(f"Type of X_train: {type(X_train)}")
        X_train = X_train.to_json(orient="split")

        X_test = load_x_y("data/X_test.pkl")
        # logger.info(f"Type of X_test: {type(X_test)}")
        X_test = X_test.to_json(orient="split")

        y_train = load_x_y("data/y_train.pkl")
        # logger.info(f"Type of y_train: {type(y_train)}")
        y_train = y_train.to_json(orient="split")

        y_test = load_x_y("data/y_test.pkl")
        # logger.info(f"Type of y_test: {type(y_test)}")
        y_test = y_test.to_json(orient="split")

        logger.info("Pickles loaded")
        return {"message": "Pickles loaded", "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}
    
@router.get("/v1/model_consult")
def model_consult(model = Depends(load_model)):
    try:
        X_test = load_x_y("data/X_test.pkl")
        y_pred = model.predict(X_test).tolist()
        logger.info("Model consulted")
        return {"message": "Model consulted", "y_pred": y_pred}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}


@router.post("/v1/predict")
def predict_churn(request: ChurnDataRequest, model = Depends(load_model)):
    try:
        request_dict = request.dict()
        prediction = get_churn_probability(request_dict, model)
        logger.info(f"Churn prediction successful: {prediction}")
        return {"churn_probability": prediction}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}
    
@router.get("/v1/features")
def get_features(model = Depends(load_model), 
                 data = Depends(load_data), 
                 X_train = Depends(lambda: load_x_y("data/X_train.pkl")), 
                 X_test = Depends(lambda: load_x_y("data/X_test.pkl"))):
    try:
        summary_fig = summary(model, data, X_train, X_test)
        logger.info("SHAP summary displayed")
        
        encoded_fig = encode_fig_to_base64(summary_fig)

        return {"message": "SHAP summary displayed", "summary_fig": encoded_fig}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}
    
@router.post("/v1/shap")
def get_shap_values(request: ShapRequest, model = Depends(load_model),
                    data = Depends(load_data),
                    X_train = Depends(lambda: load_x_y("data/X_train.pkl")), 
                    X_test = Depends(lambda: load_x_y("data/X_test.pkl"))):
    try:
        shap_plot, waterfall_plot = plot_shap(model, data, request.customer_id, X_train, X_test)
        logger.info("SHAP values displayed")
        
        # Encode the figures to base64 strings
        encoded_shap_plot = encode_fig_to_base64(shap_plot)
        encoded_waterfall_plot = encode_fig_to_base64(waterfall_plot)

        return {"message": "SHAP values displayed", "shap_plot": encoded_shap_plot, "waterfall_plot": encoded_waterfall_plot}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}