# pneumonia-xray

Classification of Pneumonia from chest X-ray images using deep learning models, CNN architecture, and transfer learning. 

Data source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

All famous pre-trained CNN architectures are used to train on X-ray images. Stacking multiple architectures are tried in order to improve accuracy.

Simple UI is also created using Streamlit. The following command will trigger the prediction code and start the interactive UI.

`streamlit run prediction_ui.py`

## Docker Usage

1. Run `docker build -t bluedata/streamlit .` to build the docker image.

2. Run `docker run -it -p 8501:8501 bluedata/streamlit` to run the container
